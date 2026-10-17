import datetime as dt
import unittest
from energy_bench import integrate, interpolate, parse_power, summarize, work_points


def events():
    rows = [{"event": "start" if i == 0 else "end" if i == 10 else "progress",
             "unix_ns": i * 1_000_000_000, "elapsed_ns": i * 1_000_000_000,
             "raw_bytes_processed": i * 2_000_000_000} for i in range(11)]
    return rows + [{"event": "verified", "roundtrip": True}]


class EnergyAccounting(unittest.TestCase):
    def test_real_macos_timezone_header(self):
        # Reduced from P8-energy-r1/preflight-power.txt (macOS build 25G83).
        text = """*** Sampled system activity (Mon Sep 21 10:54:58 2026 +0300) (1008.17ms elapsed) ***
CPU Power: 294 mW
GPU Power: 106 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 400 mW
Current pressure level: Nominal
GPU Power: 106 mW
"""
        sample = parse_power(text)[0]
        expected = dt.datetime(2026, 9, 21, 7, 54, 58, tzinfo=dt.timezone.utc).timestamp() + 0.5
        self.assertEqual(sample["end"], expected)
        self.assertAlmostEqual(sample["seconds"], 1.00817)
        self.assertAlmostEqual(sample["start"], expected - 1.00817)
        self.assertEqual(sample["cpu_W"], 0.294)
        self.assertAlmostEqual(sample["cpu_gpu_ane_W"], 0.4)
        self.assertEqual(sample["thermal_pressure"], "Nominal")
        for stamp in ("07:54:58 2026 +0000", "00:54:58 2026 -0700", "10:54:58 2026 +03:00"):
            with self.subTest(stamp=stamp):
                self.assertEqual(parse_power(text.replace("10:54:58 2026 +0300", stamp))[0]["end"], expected)
        with self.assertRaises(ValueError):
            parse_power(text.replace("+0300", "+2500"))

    def test_power_units_duration_and_unknown_counter(self):
        text = "*** Sampled system activity (Mon Sep 21 10:00:01 2026) (1250ms elapsed) ***\nCPU Power: 2000 mW\nGPU Power: 0 mW\nANE Power: 0 mW\n"
        sample = parse_power(text)[0]
        self.assertEqual(sample["seconds"], 1.25)
        self.assertEqual(sample["cpu_W"], 2)
        self.assertEqual(sample["cpu_gpu_ane_W"], 2)
        self.assertAlmostEqual(sample["end"] - sample["start"], 1.25)
        with self.assertRaises(ValueError):
            parse_power(text.replace("CPU Power: 2000 mW", "Energy Impact: 2000"))
        with self.assertRaises(ValueError):
            parse_power("sudo: a password is required")

    def test_joules_use_matching_raw_bytes_and_elapsed(self):
        samples = [{"start": i, "end": i + 1, "seconds": 1, "cpu_W": 2, "thermal_pressure": None} for i in range(10)]
        result = integrate(samples, events(), 0.5, 0.5)
        self.assertEqual(result["cpu_rail_J_per_GB"], 1)
        self.assertEqual(result["incremental_cpu_J_per_GB"], 0.75)
        self.assertTrue(result["incremental_usable"])
        self.assertEqual(result["power_samples_used"], 8)

    def test_negative_increment_is_preserved_and_not_ranked(self):
        samples = [{"start": i, "end": i + 1, "seconds": 1, "cpu_W": 2, "thermal_pressure": None} for i in range(10)]
        result = integrate(samples, events(), 3, 3)
        self.assertEqual(result["incremental_cpu_J_per_GB"], -0.5)
        self.assertFalse(result["incremental_usable"])

    def test_missing_end_verification_and_clock_jump_rejected(self):
        with self.assertRaises(ValueError):
            work_points(events()[:-1])
        broken = events()
        broken[3]["unix_ns"] += 50_000_000
        with self.assertRaises(ValueError):
            work_points(broken)
        with self.assertRaises(ValueError):
            interpolate([(0, 0), (1, 100)], 2)

    def test_null_energy_stays_null_in_summary(self):
        rows = [{"codec": c, "operation": "encode", "profile": "cpu_allocating_encode",
                 "ratio": 0.2, "GBps": 3, "cpu_rail_J_per_GB": None,
                 "cpu_rail_average_W": None, "incremental_cpu_J_per_GB": None} for c in ("vxg", "lz4-a1")]
        for row in summarize(rows):
            self.assertIsNone(row["cpu_rail_J_per_GB"])
            self.assertIsNone(row["energy_vs_lz4_a1"])


if __name__ == "__main__":
    unittest.main()
