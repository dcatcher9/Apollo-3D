"""Admission checks for the native Host mean/amplitude camera; no controller replica."""

import copy
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_coordinate_v2_contract as coordinate
import depth_coordinate_v2_dump_contract as dump
import generate_depth_coordinate_v2_contract as generator


def _float32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


class HostAdaptiveAdapterAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.contract = coordinate.load_contract()
        self.scale = _float32(coordinate.MODEL_CALIBRATIONS[0].raw_coordinate_scale)
        self.values = {field['name']: (0.0 if field['gpu_encoding'] == 'float' else 0)
                       for field in self.contract['shadow_state']['fields']}
        self.values.update(center=12.0, inverse_scale=_float32(1.0 / self.scale),
                           convergence_curve=0.0, container_scale=1.0,
                           calibration_revision=1, frame_valid=1.0,
                           contract_tag_bits=generator.contract_tag(self.contract),
                           renderer_authorization_bits=generator.contract_tag(self.contract),
                           joint_plane_mode_bits=3, gain_last_observation_low=100000,
                           gain_clock_armed=1, gain_seed_count=1,
                           gain_target_zero=12.0,
                           gain_target_inverse_scale=_float32(1.0 / self.scale),
                           gain_target_nearest=self.scale,
                           gain_display_limit=_float32(coordinate.CALIBRATED_DEFAULTS.direct_container_limit),
                           gain_seed_first_low=100000, gain_seed_last_low=100000,
                           gain_seed_mean_nearest=self.scale, gain_seed_mean_zero=12.0)

    def words(self, values=None):
        values = dict(self.values if values is None else values)
        values['camera_center_integrity_bits'] = dump.camera_center_integrity_for_state_values(values)
        return [struct.unpack('<I', struct.pack('<f', values[field['name']]))[0]
                if field['gpu_encoding'] == 'float' else values[field['name']]
                for field in self.contract['shadow_state']['fields']]

    def admit(self, values=None):
        return dump.validate_parallax_state_words(
            self.words(values), raw_coordinate_scale=self.scale,
            expected_joint_plane_mode=3, requested_gain=_float32(0.00375 * 1.75))

    def test_first_observation_authorizes_with_independent_representation_budget(self):
        result = self.admit()
        self.assertEqual(result['gain_seed_count'], 1)
        self.assertEqual(result['gain_display_limit'], _float32(0.04))
        self.assertEqual(dump.display_budget_for_mode(0.00375, 3), _float32(0.04))

    def test_camera_target_and_initial_reference_cannot_bypass_amplitude_prior(self):
        for field in ('inverse_scale', 'gain_target_nearest', 'gain_seed_mean_nearest'):
            bad = copy.deepcopy(self.values)
            bad[field] = 1.0 if field == 'inverse_scale' else self.scale * .5
            if field == 'gain_target_nearest':
                bad['gain_target_inverse_scale'] = _float32(1.0 / bad[field])
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'model prior'):
                self.admit(bad)

    def test_resealed_foreign_seed_schedule_and_display_budget_are_rejected(self):
        for field, value in (('gain_seed_count', 4), ('gain_seed_last_low', 100001),
                             ('gain_display_limit', .01), ('gain_reserved0', 1)):
            bad = copy.deepcopy(self.values)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.admit(bad)

    def test_all_controller_words_are_covered_by_the_state_seal(self):
        words = self.words()
        for index in range(12, len(words)):
            corrupt = list(words)
            corrupt[index] ^= 1
            with self.subTest(index=index), self.assertRaises(ValueError):
                dump.validate_parallax_state_words(
                    corrupt, raw_coordinate_scale=self.scale, expected_joint_plane_mode=3)

    def test_retired_modes_cannot_be_admitted_by_resealing(self):
        for mode in (0, 1, 2, 4):
            bad = dict(self.values, joint_plane_mode_bits=mode)
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'mode'):
                dump.validate_parallax_state_words(self.words(bad), raw_coordinate_scale=self.scale)

    def test_representation_cap_is_independent_of_requested_strength(self):
        for requested_gain in (0.001, 0.0065625, 0.03):
            result = dump.validate_parallax_state_words(
                self.words(), raw_coordinate_scale=self.scale,
                expected_joint_plane_mode=3, requested_gain=requested_gain)
            self.assertEqual(result['gain_display_limit'], _float32(0.04))
        bad = dict(self.values, gain_display_limit=_float32(0.01))
        with self.assertRaisesRegex(ValueError, 'representation budget'):
            dump.validate_parallax_state_words(
                self.words(bad), raw_coordinate_scale=self.scale, expected_joint_plane_mode=3)


if __name__ == '__main__':
    unittest.main()
