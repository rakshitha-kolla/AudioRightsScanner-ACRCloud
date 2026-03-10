import requests
import base64
import hmac
import hashlib
from datetime import datetime
import subprocess
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
logger = logging.getLogger(__name__)

DEFAULT_MERGE_GAP = 5.0


class ACRCloudService:

    def __init__(self, access_key, access_secret, host='identify-us.acrcloud.com'):
        self.access_key = access_key
        self.access_secret = access_secret
        self.host = host

    def _trim_audio(self, audio_file_path, duration_seconds=15):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
                tmp_path = tmp.name
            cmd = ['ffmpeg', '-i', audio_file_path, '-t', str(duration_seconds),
                   '-q:a', '9', '-acodec', 'libmp3lame', tmp_path, '-y']
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg error: {result.stderr.decode()}")
            with open(tmp_path, 'rb') as f:
                return f.read()
        except Exception:
            with open(audio_file_path, 'rb') as f:
                return f.read()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _extract_chunk(self, audio_file_path, start_sec, end_sec):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
            chunk_path = tmp.name
        cmd = ['ffmpeg', '-i', audio_file_path,
               '-ss', str(start_sec), '-t', str(end_sec - start_sec),
               '-acodec', 'libmp3lame', '-q:a', '9', chunk_path, '-y']
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            if os.path.exists(chunk_path):
                os.unlink(chunk_path)
            raise RuntimeError(
                f"ffmpeg failed extracting {start_sec}-{end_sec}s: "
                f"{result.stderr.decode(errors='replace')}"
            )
        return chunk_path

    def _get_probe_points(self, seg_start, seg_end, probe_interval=8.0):
        points = []
        current = seg_start
        while current < seg_end - 2.0:
            points.append(current)
            current += probe_interval
        return points

    def identify(self, audio_file_path):
        try:
            audio_data = self._trim_audio(audio_file_path, duration_seconds=15)
            timestamp = str(int(datetime.now().timestamp()))
            signature_string = f"POST\n/v1/identify\n{self.access_key}\naudio\n1\n{timestamp}"
            signature = base64.b64encode(
                hmac.new(self.access_secret.encode(), signature_string.encode(), hashlib.sha1).digest()
            ).decode()
            files = {
                'sample': audio_data,
                'access_key': (None, self.access_key),
                'timestamp': (None, timestamp),
                'signature': (None, signature),
                'data_type': (None, 'audio'),
                'signature_version': (None, '1'),
            }
            response = requests.post(f'https://{self.host}/v1/identify', files=files, timeout=60)
            return self._parse_result(response.json())
        except FileNotFoundError:
            return {'copyrighted': None, 'error': 'File not found'}
        except requests.exceptions.RequestException as e:
            return {'copyrighted': None, 'error': f'Network error: {str(e)}'}
        except Exception as e:
            return {'copyrighted': None, 'error': f'Error: {str(e)}'}

    def _parse_result(self, result):
        status_block = result.get('status') or result
        code = status_block.get('code')
        if code == 0:
            music_list = result.get('metadata', {}).get('music', [])
            if music_list:
                music = music_list[0]
                return {
                    'copyrighted': True,
                    'music': {
                        'title': music.get('title'),
                        'artist': music['artists'][0]['name'] if music.get('artists') else 'Unknown',
                        'album': music.get('album', {}).get('name', 'Unknown'),
                        'duration': music.get('duration_ms', 0) // 1000,
                        'acrid': music.get('acrid'),
                    }
                }
        elif code == 1:
            return {'copyrighted': False}
        else:
            msg = status_block.get('msg') or result.get('message', 'Unknown error')
            return {'copyrighted': None, 'error': f'API Error ({code}): {msg}'}
        return {'copyrighted': None, 'error': f'Invalid response: {result}'}

    def _probe_single_point(self, audio_file_path, probe_start, probe_end):
        chunk_path = self._extract_chunk(audio_file_path, probe_start, probe_end)
        try:
            result = self.identify(chunk_path)
        finally:
            if os.path.exists(chunk_path):
                os.unlink(chunk_path)
        return probe_start, probe_end, result

    def identify_with_yamnet(self, audio_file_path,
                              confidence_threshold=0.1,
                              min_segment_duration=3.0,
                              chroma_threshold=0.35,
                              detector_instance=None,
                              max_workers=4):
        from .yamnet_detector import YAMNetDetector
        try:
            if detector_instance is not None:
                detector = detector_instance.clone()
                detector.confidence_threshold = confidence_threshold
                detector.min_segment_duration = min_segment_duration
            else:
                detector = YAMNetDetector(
                    confidence_threshold=confidence_threshold,
                    min_segment_duration=min_segment_duration
                )

            candidate_segments = detector.get_music_segments(audio_file_path)
            if not candidate_segments:
                return {"copyrighted": False, "segments": []}
            all_probes = []
            for seg in candidate_segments:
                start, end = seg["start"], seg["end"]
                if end - start < 2.0:
                    continue
                for probe_start in self._get_probe_points(start, end, probe_interval=8.0):
                    probe_end = min(probe_start + 12.0, end)
                    all_probes.append((probe_start, probe_end))

            if not all_probes:
                return {"copyrighted": False, "segments": []}
            probe_results = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_probe = {
                    executor.submit(
                        self._probe_single_point, audio_file_path, ps, pe
                    ): (ps, pe)
                    for ps, pe in all_probes
                }
                for future in as_completed(future_to_probe):
                    ps, pe = future_to_probe[future]
                    try:
                        probe_start, probe_end, result = future.result()
                        probe_results[probe_start] = (probe_end, result)
                    except Exception as e:
                        logger.warning(f"Probe {ps}-{pe}s failed: {e}")
                        probe_results[ps] = (pe, {'copyrighted': None, 'error': str(e)})

            confirmed_segments = []
            for probe_start in sorted(probe_results.keys()):
                probe_end, result = probe_results[probe_start]
                if not result.get("copyrighted"):
                    continue
                music = result["music"]
                acrid = music.get("acrid")
                prev_acrid = (
                    confirmed_segments[-1].get("music", {}).get("acrid")
                    if confirmed_segments else None
                )
                if confirmed_segments and acrid and acrid == prev_acrid:
                    confirmed_segments[-1]["end"] = round(probe_end, 2)
                    confirmed_segments[-1]["duration"] = round(
                        probe_end - confirmed_segments[-1]["start"], 2
                    )
                else:
                    confirmed_segments.append({
                        "start": round(probe_start, 2),
                        "end": round(probe_end, 2),
                        "duration": round(probe_end - probe_start, 2),
                        "music": music,
                    })

            return {"copyrighted": len(confirmed_segments) > 0, "segments": confirmed_segments}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"copyrighted": None, "error": str(e)}

    def identify_with_timeline(self, audio_file_path, chunk_seconds=10, overlap_seconds=2,max_workers=4):
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1", audio_file_path]
            probe = subprocess.run(cmd, capture_output=True, text=True)
            total_duration = float(probe.stdout.strip())
            step = chunk_seconds - overlap_seconds
            current = 0.0
            probe_windows = []
            while current < total_duration:
                probe_windows.append((current, min(current + chunk_seconds, total_duration)))
                current += step
            probe_results = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_window = {
                    executor.submit(
                        self._probe_single_point, audio_file_path, start, end
                    ): (start, end)
                    for start, end in probe_windows
                }
                for future in as_completed(future_to_window):
                    start, end = future_to_window[future]
                    try:
                        ps, pe, result = future.result()
                        probe_results[ps] = (pe, result)
                    except Exception as e:
                        logger.warning(f"Timeline probe {start}-{end}s failed: {e}")

            segments = []
            for start in sorted(probe_results.keys()):
                end, result = probe_results[start]
                if result.get("copyrighted"):
                    segments.append({"start": start, "end": end, "music": result.get("music")})

            merged = self.merge_overlapping_segments(segments)
            return {"copyrighted": len(merged) > 0, "segments": merged}
        except Exception as e:
            return {"copyrighted": None, "error": str(e)}

    def merge_overlapping_segments(self,segments,gap_threshold=DEFAULT_MERGE_GAP):
        if not segments:
            return []

        segments.sort(key=lambda x: x["start"])
        merged = [segments[0]]
        for current in segments[1:]:
            last = merged[-1]
            time_close = current["start"] - last["end"] <= gap_threshold
            current_music = current.get("music", {})
            last_music = last.get("music", {})

            current_acrid = current_music.get("acrid")
            last_acrid = last_music.get("acrid")

            same_acrid = current_acrid and last_acrid and current_acrid == last_acrid
            same_artist = current_music.get("artist") == last_music.get("artist")
            same_title = (
                current_music.get("title", "").strip().lower()
                == last_music.get("title", "").strip().lower()
            )

            if time_close and (same_acrid or (same_artist and same_title)):
                last["end"] = max(last["end"], current["end"])
                last["duration"] = round(last["end"] - last["start"], 2)
            else:
                merged.append(current)
        return merged