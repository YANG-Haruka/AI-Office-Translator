from pipeline.epub_translation_pipeline import extract_epub_content_to_json, write_translated_content_to_epub
from .base_translator import DocumentTranslator


class EpubTranslator(DocumentTranslator):
    def extract_content_to_json(self, progress_callback=None):
        return extract_epub_content_to_json(self.input_file_path)

    def write_translated_json_to_file(self, json_path, translated_json_path, progress_callback=None):
        write_translated_content_to_epub(self.input_file_path, json_path, translated_json_path)
