import tempfile
import unittest

import numpy as np
import pandas as pd

from process_tess_targets import (
    TIC_COLUMNS, categorize_targets, clean_dataset,
    impute_stellar_parameters, iter_tic_csv, process_local_csv, read_tic_csv,
)


class TessPipelineTests(unittest.TestCase):
    def test_reads_headerless_bulk_schema_in_compact_mode(self):
        row = [''] * len(TIC_COLUMNS)
        row[TIC_COLUMNS.index('ID')] = '42'
        row[TIC_COLUMNS.index('objType')] = 'STAR'
        row[TIC_COLUMNS.index('Tmag')] = '10.5'
        with tempfile.NamedTemporaryFile('w', suffix='.csv') as handle:
            handle.write(','.join(row) + '\n')
            handle.flush()
            frame = read_tic_csv(handle.name, compact=True)
        self.assertEqual(frame.loc[0, 'ID'], 42)
        expected = set(__import__('process_tess_targets').COMPACT_COLUMNS) & set(TIC_COLUMNS)
        self.assertEqual(set(frame.columns), expected)

    def test_compact_mode_preserves_optional_ruwe_and_feh(self):
        frame = pd.DataFrame({
            'ID': [42], 'objType': ['STAR'], 'ruwe': [1.1], 'feh': [0.2],
            'contratio': [0.03], 'unneeded_column': ['discard me'],
        })
        with tempfile.NamedTemporaryFile('w', suffix='.csv') as handle:
            frame.to_csv(handle.name, index=False)
            result = read_tic_csv(handle.name, compact=True)
        self.assertIn('ruwe', result.columns)
        self.assertIn('feh', result.columns)
        self.assertIn('MH', result.columns)
        self.assertNotIn('unneeded_column', result.columns)
        self.assertEqual(result.loc[0, 'MH'], 0.2)

    def test_large_csv_can_be_streamed_in_chunks(self):
        frame = pd.DataFrame({'ID': [1, 2, 3], 'objType': ['STAR'] * 3})
        with tempfile.NamedTemporaryFile('w', suffix='.csv') as handle:
            frame.to_csv(handle.name, index=False)
            chunks = list(iter_tic_csv(handle.name, compact=True, chunk_size=2))
        self.assertEqual([len(chunk) for chunk in chunks], [2, 1])

    def test_complete_output_preserves_original_columns(self):
        frame = pd.DataFrame({
            'ID': [1], 'objType': ['STAR'], 'contratio': [0.01],
            'custom_original_column': ['preserved'], 'Tmag': [10.0],
        })
        with tempfile.NamedTemporaryFile('w', suffix='.csv') as source, \
                tempfile.NamedTemporaryFile('w', suffix='.csv') as output:
            frame.to_csv(source.name, index=False)
            process_local_csv(source.name, output.name, compact=False, chunk_size=1,
                              filters=['high_fidelity_photon'], categorize=True)
            result = pd.read_csv(output.name)
        self.assertEqual(result.loc[0, 'custom_original_column'], 'preserved')
        self.assertTrue(result.loc[0, 'is_high_fidelity_photon'])

    def test_cleaning_applies_available_quality_cuts(self):
        frame = pd.DataFrame({
            'objType': ['STAR', 'GALAXY', 'STAR'],
            'disposition': [np.nan, np.nan, 'ARTIFACT'],
            'contratio': [0.05, 0.01, 0.01],
        })
        self.assertEqual(len(clean_dataset(frame)), 1)

    def test_imputation_is_vectorized_and_physical(self):
        frame = pd.DataFrame({'rad': [0.5, 2.0], 'mass': [np.nan, np.nan]})
        result = impute_stellar_parameters(frame)
        np.testing.assert_allclose(result.mass, [0.5 ** 1.25, 2.0 ** 1.75])

    def test_categories_are_independent_not_intersected(self):
        frame = pd.DataFrame({
            'Tmag': [10.0, 13.0], 'contratio': [0.2, 0.01],
        })
        result = categorize_targets(frame, ['high_fidelity_photon', 'aperture_purity'])
        self.assertEqual(result.selection_count.tolist(), [1, 1])
        self.assertEqual(result.selection_labels.tolist(), ['high_fidelity_photon', 'aperture_purity'])


if __name__ == '__main__':
    unittest.main()
