import json
from typing import List

class DataReader:
    def __init__(self, data_path: str, ref_path: str):
        self.data_path = data_path
        self.ref_path = ref_path
        self.current_index = 0
        self._read_data_from_json()

    def _read_data_from_json(self):
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.data = data["samples"]

            with open(self.ref_path, 'r', encoding='utf-8') as f:
                ref = json.load(f)
            self.ref = ref
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Error loading data: {e}")

    def get_corpus(self) -> List[str]:
        """
        Extract reference texts from ref data to create a corpus for RAG model.
        Returns a list of reference strings that can be used as corpus.
        """
        corpus = []
        for item in self.ref:
            if isinstance(item, dict) and 'reference' in item:
                corpus.append(item['reference'])
        return corpus

    def get_ref_by_sample_id(self, sample_id: str) -> str:
        """
        Get reference text for a specific sample_id.
        """
        for item in self.ref:
            if isinstance(item, dict) and item.get('sample_id') == sample_id:
                return item.get('reference', '')
        return ''

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self.data):
            raise StopIteration
        sample = self.data[self.current_index]
        self.current_index += 1
        return sample

    