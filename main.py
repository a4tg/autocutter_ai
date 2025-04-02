import os
import asyncio
import re
import ffmpeg
import cv2
import numpy as np
import whisper
import librosa
import logging
import random
from ruaccent import RUAccent
from moviepy import CompositeVideoClip, VideoFileClip
from transformers import pipeline
from keybert import KeyBERT
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    Doc
)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Пути и директории
video_file_path = "C:/Users/S1NTET1KA/Desktop/Work_coding/input.mp4"
base_output_folder = "D:/Сutted"
os.makedirs(base_output_folder, exist_ok=True)

# Константы
MIN_DURATION = 30  # Минимальная длительность клипа (секунды)
MAX_DURATION = 45  # Максимальная длительность клипа (секунды)
MIN_CLIPS = 50 # Минимальное количество клипов
MAX_CLIPS = 60 # Максимальное количество клипов

# Инициализация моделей
accentizer = RUAccent()
accentizer.load(omograph_model_size='turbo', use_dictionary=True)
whisper_model = whisper.load_model("small", device="cuda")
sentiment_analyzer = pipeline(
    "text-classification",
    model="cointegrated/rubert-tiny2-cedr-emotion-detection",
    device=0
)
kw_model = KeyBERT(model="paraphrase-multilingual-mpnet-base-v2")

# Инициализация Natasha
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
ner_tagger = NewsNERTagger(emb)


def postprocess_text(text):
    try:
        processed_text = accentizer.process_all(text)
        return processed_text.replace("+", "")
    except Exception as e:
        logging.error(f"Ошибка постобработки текста: {str(e)}")
        return text


def split_into_sentences(text):
    doc = Doc(text)
    doc.segment(segmenter)
    return [sent.text for sent in doc.sents]


def detect_speech_pauses(audio_path, pause_threshold=0.05, min_pause_duration=0.3):
    y, sr = librosa.load(audio_path, sr=None)
    intervals = librosa.effects.split(y, top_db=40, frame_length=4096, hop_length=1024)
    return [(start / sr, end / sr) for start, end in intervals]


def adjust_segment_boundaries(segments, pauses, min_duration=30, max_duration=45, buffer=1.0):
    adjusted_segments = []

    for seg in segments:
        start = seg['start']
        end = seg['end']

        # Коррекция начала клипа
        for pause_start, pause_end in pauses:
            if pause_end <= start and (start - pause_end) <= buffer:
                start = pause_end
                break

        # Коррекция конца клипа
        for pause_start, pause_end in pauses:
            if pause_start >= end and (pause_start - end) <= buffer:
                end = pause_start
                break

        # Проверяем длительность сегмента
        duration = end - start
        if duration < min_duration:
            end = min(start + min_duration, segments[-1]['end'])  # Не выходить за последний сегмент
        elif duration > max_duration:
            end = start + max_duration  # Обрезаем длинные сегменты

        adjusted_segments.append({
            'start': start,
            'end': end,
            'text': seg['text']
        })

    return adjusted_segments


def adjust_clip_boundaries(clip_start, clip_end, pauses, audio_duration):
    # Используем исходные границы, выбранные итеративным выбором
    prev_pauses = [p for p in pauses if p[1] <= clip_start]
    next_pauses = [p for p in pauses if p[0] >= clip_end]
    new_start = prev_pauses[-1][1] if prev_pauses else clip_start
    new_end = next_pauses[0][0] if next_pauses else clip_end

    if new_end - new_start < MIN_DURATION:
        new_end = min(new_start + MIN_DURATION + random.uniform(0.1, 0.5), audio_duration)
    if (new_end - new_start) > MAX_DURATION:
        new_end = new_start + MAX_DURATION
    new_end = min(new_end, audio_duration)
    logging.debug(f"Корректировка границ: вход ({clip_start}, {clip_end}), выход ({new_start}, {new_end})")
    return max(0, new_start), new_end


def group_segments(segments, transcript):
    sentences = split_into_sentences(transcript)
    sentence_segments = []
    current_sentence = ""
    current_start = None
    current_end = 0
    for seg in sorted(segments, key=lambda x: x['start']):
        if not current_sentence:
            current_start = seg['start']
        current_sentence = (current_sentence + " " + seg['text']).strip()
        current_end = seg['end']
        for sent in sentences:
            if sent.strip() in current_sentence and current_sentence[-1] in ".?!":
                sentence_segments.append({
                    'start': current_start,
                    'end': current_end,
                    'text': postprocess_text(sent.strip())
                })
                current_sentence = ""
                current_start = None
                break
    if current_sentence:
        sentence_segments.append({
            'start': current_start,
            'end': current_end,
            'text': postprocess_text(current_sentence.strip())
        })
    return sentence_segments


def analyze_text(text):
    sentiment = sentiment_analyzer(text[:512])[0]
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="russian")
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.tag_ner(ner_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    names = [span.text for span in doc.spans if span.type == "PER"]
    locations = [span.text for span in doc.spans if span.type == "LOC"]
    return {
        "sentiment": sentiment["label"],
        "confidence": sentiment["score"],
        "keywords": [kw[0] for kw in keywords],
        "entities": {"names": names, "locations": locations}
    }


def score_candidate(candidate):
    analysis = analyze_text(candidate["text"])
    weight = analysis["confidence"]
    doc = Doc(candidate["text"])
    doc.segment(segmenter)
    if len(doc.sents) == 1:
        weight *= 1.5
    if analysis["sentiment"] != "neutral":
        weight *= 1.5
    weight *= 1 + 0.1 * len(analysis["keywords"])
    weight *= 1 + 0.2 * len(analysis["entities"]["names"])
    weight *= 1 + 0.1 * len(analysis["entities"]["locations"])
    duration = candidate["end"] - candidate["start"]
    if duration <= MAX_DURATION:
        weight *= 1.2
    return weight * len(candidate["text"].split())


def text_preprocessing(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def time_overlap_percentage(a_start, a_end, b_start, b_end):
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    if overlap_start >= overlap_end:
        return 0.0
    overlap_duration = overlap_end - overlap_start
    total_duration = min(a_end - a_start, b_end - b_start)
    return overlap_duration / total_duration if total_duration > 0 else 0


def deduplicate_candidates(candidates, text_similarity_threshold=0.85, time_overlap_threshold=0.7, time_proximity_threshold=5.0, tol=0.1):
    deduped = []
    candidates = sorted(candidates, key=lambda x: x["start"])
    for candidate in candidates:
        candidate_text = text_preprocessing(candidate["text"])
        candidate_start = candidate["start"]
        candidate_end = candidate["end"]
        duplicate = False
        for existing in deduped:
            if abs(candidate_start - existing["start"]) < tol and abs(candidate_end - existing["end"]) < tol:
                duplicate = True
                break
        if not duplicate:
            deduped.append(candidate)
    return deduped


def iterative_candidate_selection(segments, video_duration, max_clips=MAX_CLIPS):
    # Сначала убираем дубликаты
    remaining = deduplicate_candidates(segments)
    remaining = sorted(remaining, key=lambda x: x['start'])
    selected = []
    while remaining and len(selected) < max_clips:
        candidate = max(remaining, key=score_candidate)
        selected.append(candidate)
        cand_start, cand_end = candidate['start'], candidate['end']
        logging.info(f"Выбран кандидат: {cand_start:.2f} - {cand_end:.2f}")
        new_remaining = []
        for seg in remaining:
            # Удаляем сегмент, если он хоть частично пересекается с выбранным кандидатом
            if seg['end'] <= cand_start or seg['start'] >= cand_end:
                new_remaining.append(seg)
            else:
                logging.debug(f"Удаляем сегмент {seg['start']:.2f}-{seg['end']:.2f} из-за перекрытия с кандидатом {cand_start:.2f}-{cand_end:.2f}")
        remaining = new_remaining
    return selected


def sequential_candidate_selection(segments, video_duration, min_clips=MIN_CLIPS, max_clips=MAX_CLIPS):
    sequential_candidates = []
    current_time = 0
    segments_sorted = sorted(segments, key=lambda x: x['start'])
    while current_time < video_duration and len(sequential_candidates) < max_clips:
        available = [seg for seg in segments_sorted if seg['start'] >= current_time]
        if not available:
            break
        candidate = max(available, key=score_candidate)
        sequential_candidates.append(candidate)
        logging.debug(f"Выбран кандидат: {candidate['start']} - {candidate['end']}")
        current_time = candidate['end'] + 0.5
    return sequential_candidates


def extract_audio(video_path, audio_path="temp_audio.wav"):
    try:
        ffmpeg.input(video_path).output(audio_path, acodec="pcm_s16le", ac=1, ar=16000).run(
            overwrite_output=True, capture_stdout=True, capture_stderr=True)
        return audio_path
    except ffmpeg.Error as e:
        logging.error(f"Ошибка извлечения аудио: {e.stderr.decode()}")
        return None


async def transcribe_audio(audio_path):
    try:
        result = whisper_model.transcribe(audio_path, language="russian")
        return result.get("text", ""), result.get("segments", [])
    except Exception as e:
        logging.error(f"Ошибка транскрибации: {str(e)}")
        return "", []


def create_clip(video_path, start_time, end_time, output_folder, clip_index):
    duration = end_time - start_time
    if duration > MAX_DURATION:
        logging.warning(f"Клип превышает MAX_DURATION ({duration:.1f}s), но сохранен для целостности")
    try:
        output_path = os.path.join(output_folder, f"clip_{clip_index}.mp4")
        logging.info(f"Создается клип {clip_index}: {start_time:.2f} - {end_time:.2f} (duration: {duration:.2f})")
        ffmpeg.input(video_path, ss=start_time, t=duration).output(
            output_path, vcodec='libx264', acodec='copy', crf=22
        ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
        return output_path
    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr.decode()}")
        return None


def blur_background_ffmpeg(input_path, output_path, final_width=1080, final_height=1920, blur_strength=10):
    """
    Масштабирует видео до формата 9:16, затем размывает его с помощью фильтра gblur.
    """
    try:
        (
            ffmpeg
            .input(input_path)
            .filter('scale', final_width, final_height)
            .filter('gblur', sigma=blur_strength)
            .output(output_path, vcodec='libx264', pix_fmt='yuv420p', acodec='copy')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return True
    except ffmpeg.Error as e:
        logging.error(f"Ошибка при создании размытого фона: {e.stderr.decode()}")
        return False


def compose_final_video(clip_path, background_path, output_path, final_width=1080, final_height=1920):
    try:
        # Базовые размеры обрезки
        desired_crop_width = 1536
        desired_crop_height = 1080

        # Получаем размеры оригинального видео
        probe = ffmpeg.probe(clip_path)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        orig_width = int(video_stream['width'])
        orig_height = int(video_stream['height'])

        # Проверка: если видео слишком маленькое — ошибка
        if orig_width < 100 or orig_height < 100:
            logging.error(f"Видео слишком маленькое: {orig_width}x{orig_height}")
            return False

        # Адаптивная обрезка, если видео меньше нужного
        crop_width = min(orig_width, desired_crop_width)
        crop_height = min(orig_height, desired_crop_height)

        # Смещения для обрезки по центру
        x_offset = max((orig_width - crop_width) // 2, 0)
        y_offset_crop = max((orig_height - crop_height) // 2, 0)

        # Высота переднего плана после масштабирования до 1080 по ширине
        final_foreground_height = int((final_width * crop_height) / crop_width)

        # Центрируем по вертикали
        y_offset_overlay = (final_height - final_foreground_height) // 2

        # Передний план: обрезка, масштабирование
        foreground_video = (
            ffmpeg
            .input(clip_path)
            .video
            .filter("crop", crop_width, crop_height, x_offset, y_offset_crop)
            .filter("scale", final_width, final_foreground_height)
            .filter("setsar", "1")
            .filter("setpts", "PTS-STARTPTS")
        )

        # Получаем аудиопоток из исходного видео
        audio = ffmpeg.input(clip_path).audio

        # Фон: уже масштабированный и размытый 1080x1920
        background = (
            ffmpeg
            .input(background_path)
            .video
            .filter("setsar", "1")
            .filter("setpts", "PTS-STARTPTS")
        )

        # Сведение в одно видео
        composed = ffmpeg.overlay(background, foreground_video, x=0, y=y_offset_overlay)

        (
            ffmpeg
            .output(composed, audio, output_path,
                   vcodec='libx264', pix_fmt='yuv420p',
                   acodec='aac', shortest=None)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        return True

    except ffmpeg.Error as e:
        logging.error(f"Ошибка при компоновке видео: {e.stderr.decode()}")
        return False



async def process_video_pipeline(video_path):
    try:
        user_folder_name = input("Введите имя папки для сохранения нарезок: ").strip()
        base_clip_folder = os.path.join(base_output_folder, user_folder_name)
        os.makedirs(base_clip_folder, exist_ok=True)
        clip_dir = os.path.join(base_clip_folder, "clip")
        resized_dir = os.path.join(base_clip_folder, "resized")
        final_dir = os.path.join(base_clip_folder, "final")
        os.makedirs(clip_dir, exist_ok=True)
        os.makedirs(resized_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)

        logging.info("Извлекаем аудио...")
        audio_path = extract_audio(video_path)
        if not audio_path:
            raise RuntimeError("Не удалось извлечь аудио")

        logging.info("Транскрибация...")
        transcript, segments = await transcribe_audio(audio_path)
        if not segments:
            raise RuntimeError("Транскрибация не дала результатов")

        logging.info("Анализ пауз...")
        pauses = detect_speech_pauses(audio_path)
        logging.info("Группировка сегментов...")
        grouped_segments = group_segments(segments, transcript)
        if not grouped_segments:
            raise RuntimeError("Не удалось сгруппировать сегменты")
        logging.info("Корректировка границ сегментов...")
        adjusted_segments = adjust_segment_boundaries(grouped_segments, pauses)

        probe = ffmpeg.probe(video_path)
        video_duration = float(probe['format']['duration'])
        logging.info(f"Длительность видео: {video_duration:.2f} секунд")
        logging.info("Итеративный выбор кандидатов...")
        selected = iterative_candidate_selection(adjusted_segments, video_duration)
        if not selected:
            raise RuntimeError("Не найдено подходящих кандидатов")
        selected.sort(key=lambda x: x["start"])
        logging.info(f"Выбранные клипы: {[(c['start'], c['end']) for c in selected]}")

        final_outputs = []
        for clip_index, candidate in enumerate(selected):
            try:
                orig_start, orig_end = adjust_clip_boundaries(candidate["start"], candidate["end"], pauses, video_duration)
                logging.info(f"Создание клипа {clip_index}: {orig_start:.2f} - {orig_end:.2f}")
                clip_path = create_clip(video_path, orig_start, orig_end, clip_dir, clip_index)
                if not clip_path:
                    logging.error(f"Ошибка создания клипа для кандидата: {candidate}")
                    continue

                # Генерация размытого фона
                resized_path = os.path.join(resized_dir, f"resized_{clip_index}.mp4")
                logging.info(f"Генерация размытого фона для клипа: {clip_path}")
                if not blur_background_ffmpeg(clip_path, resized_path, final_width=1080, final_height=1920):
                    logging.error(f"Ошибка генерации размытого фона для клипа: {clip_path}")
                    continue

                # Составление финального видео с наложением оригинального клипа на размытый фон
                final_path = os.path.join(final_dir, f"final_{clip_index}.mp4")
                if compose_final_video(clip_path, resized_path, final_path, final_width=1080, final_height=1920):
                    final_outputs.append(final_path)
                    os.remove(resized_path)  # Удаляем временный файл размытого фона
                else:
                    logging.error(f"Ошибка при создании финального видео для {resized_path}")
            except Exception as e:
                logging.error(f"Ошибка обработки клипа для кандидата {candidate}: {str(e)}", exc_info=True)

        logging.info(f"Успешно обработано клипов: {len(final_outputs)}")
        return final_outputs
    except Exception as e:
        logging.error(f"Критическая ошибка: {str(e)}", exc_info=True)
        return []

if __name__ == "__main__":
    asyncio.run(process_video_pipeline(video_file_path))
