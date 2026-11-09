"""Tests for identically keyed, nested inputs and output-only alterations."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import length_workload as w
from workload import row, write_rows


class Words:
    """Reversible token-per-word tokenizer with three chat-template tokens."""
    def __init__(self):
        self.to_id = {}
        self.to_word = {}

    def encode(self, text, add_special_tokens=False):
        result = []
        for word in text.split():
            value = self.to_id.get(word)
            if value is None:
                value = self.to_id[word] = 3 + len(self.to_id)
                self.to_word[value] = word
            result.append(value)
        return result

    def decode(self, ids, skip_special_tokens=False):
        return ' '.join(self.to_word[i] for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        return [0] + self.encode(messages[0]['content']) + [1, 2]


def make_anchor(anchor, tokenizer):
    # Borrow the archived ID and document-arrival schedule so all counts and
    # positions match the deployed protocol. Text is synthetic for this CPU test.
    reference = w.ANCHOR / 'reuse8_seed33962'
    directory = anchor / reference.name
    directory.mkdir(parents=True)
    case = w._load(w.ANCHOR / 'original_case.json')
    docs = {}
    for phase, filename in w.FILENAMES.items():
        original = (w._load if phase == 'smoke' else w._lines)(reference / filename)
        items = []
        for x in original:
            key = x['key']
            if key not in docs:
                docs[key] = 'document-' + str(len(docs)) + (' z' * 7935)
            items.append(row(tokenizer, docs[key], x['id'], 8192, 32, 7936))
        if phase == 'smoke':
            w._dump(directory / filename, items)
        else:
            case['files'][phase] = write_rows(directory / filename, items)
    w._dump(anchor / 'original_case.json', case)
    calibration = anchor / 'calibration'
    calibration.mkdir()
    for filename in w.CALIBRATION:
        write_rows(calibration / filename, [row(tokenizer, 'calibration z', filename, 2048, 32, 1536)])
    paths = sorted(p for p in anchor.rglob('*') if p.is_file())
    w._dump(anchor / 'source_hashes.json', {str(p.relative_to(anchor)): w._sha(p) for p in paths})


class LengthWorkloadTests(unittest.TestCase):
    def test_shipped_archive_has_original_counts_and_file_hashes(self):
        hashes = w.verify_anchor_files()
        self.assertEqual(len(hashes), 8)
        self.assertEqual(hashes['reuse8_seed33962/measured.jsonl'],
                         'e579aa2d97f1323627a0d8a2091970f07afb812d754cb730d35a557767d087ea')

    def test_tokenizer_drift_stops_before_output(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / 'workloads'
            with self.assertRaisesRegex(ValueError, 'tokenizer does not reproduce'):
                w.build(Words(), out)
            self.assertFalse(out.exists())

    def test_matched_workloads_full_grid(self):
        with TemporaryDirectory() as tmp:
            anchor, out = Path(tmp) / 'anchor', Path(tmp) / 'workloads'
            tokenizer = Words()
            make_anchor(anchor, tokenizer)
            manifest = w.build(tokenizer, out, anchor)
            self.assertEqual(len(manifest['cells']), 6)
            self.assertEqual(sum(c['files']['measured']['count'] + c['files']['learn']['count']
                                 for c in manifest['cells']), 1152)
            for phase, filename in w.FILENAMES.items():
                load = w._load if phase == 'smoke' else w._lines
                original_file = anchor / 'reuse8_seed33962' / filename
                source = load(original_file)
                self.assertEqual((out / 'p8192_o32_seed33962' / filename).read_bytes(),
                                 original_file.read_bytes())
                for size in w.INPUTS:
                    a = load(out / f'p{size}_o32_seed33962' / filename)
                    b = load(out / f'p{size}_o128_seed33962' / filename)
                    self.assertEqual([r['id'] for r in a], [r['id'] for r in source])
                    self.assertEqual([r['key'] for r in a], [r['key'] for r in source])
                    self.assertEqual([r['first_block_digest'] for r in a],
                                     [r['first_block_digest'] for r in source])
                    self.assertEqual([w._with_output(r, 128) for r in a], b)
                    self.assertEqual(len({r['prompt_sha256'] for r in a}), len(a))
                    # Check at first and last request for both changed lengths.
                    for r in (a[0], a[-1], b[0], b[-1]):
                        self.assertEqual(len(w._actual_ids(tokenizer, r)), size)
            for name in w.CALIBRATION:
                self.assertEqual((anchor / 'calibration' / name).read_bytes(),
                                 (out / 'calibration' / name).read_bytes())
            with self.assertRaises(FileExistsError):
                w.build(tokenizer, out, anchor)
            receipt = w._load(out / 'length_match_receipt.json')
            self.assertTrue(receipt['nested_prefix_tokens_checked'])
            self.assertEqual(set(receipt['input_cohort_hashes']), {'4096', '6144', '8192'})


if __name__ == '__main__':
    unittest.main()
