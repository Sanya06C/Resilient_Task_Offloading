import unittest

from eh_extension.battery import BatteryAccount, BatteryConfig


class BatteryAccountTests(unittest.TestCase):
    def test_observed_sign_convention_and_capacity(self):
        battery = BatteryAccount(BatteryConfig(capacity_mah=100.0, initial_soc=0.5))
        charged = battery.step_terminal_current(timestamp=0, duration_s=3600, battery_current_ma=-20)
        self.assertAlmostEqual(charged.soc_after, 0.7)
        discharged = battery.step_terminal_current(timestamp=1, duration_s=3600, battery_current_ma=30)
        self.assertAlmostEqual(discharged.soc_after, 0.4)

    def test_external_balance_preserves_separate_efficiencies(self):
        battery = BatteryAccount(BatteryConfig(
            capacity_mah=100.0, initial_soc=0.5, charge_efficiency=0.5,
            discharge_efficiency=0.5,
        ))
        result = battery.step_external_balance(
            timestamp=0, duration_s=3600, harvest_charge_current_ma=20,
            action_discharge_current_ma=20,
        )
        # 20 mAh arrives, 10 mAh stores; 20 mAh load requires 40 mAh storage.
        self.assertAlmostEqual(result.soc_after, 0.2)

    def test_empty_battery_reports_unmet_demand(self):
        battery = BatteryAccount(BatteryConfig(capacity_mah=10.0, initial_soc=0.0))
        result = battery.step_terminal_current(timestamp=0, duration_s=3600, battery_current_ma=2)
        self.assertAlmostEqual(result.unmet_discharge_mah, 2.0)
        self.assertEqual(result.soc_after, 0.0)


if __name__ == "__main__":
    unittest.main()
