import unittest

import _bootstrap  # noqa: F401
from parsers import bgl, openssh


class BGLParserTests(unittest.TestCase):
    def test_parses_empty_fatal_content(self):
        line = (
            "- 1120866517 2005.07.08 R22-M0-N0-C:J08-U11 "
            "2005-07-08-16.48.37.817558 R22-M0-N0-C:J08-U11 "
            "RAS KERNEL FATAL"
        )
        record = bgl.parse_line(line, 7)

        self.assertIsNotNone(record)
        self.assertEqual(record["line_id"], 7)
        self.assertEqual(record["is_anomaly"], 0)
        self.assertEqual(record["level"], "FATAL")
        self.assertEqual(record["content"], "<EMPTY>")

    def test_parses_labeled_failure_as_anomaly(self):
        line = (
            "APPREAD 1120866517 2005.07.08 R22-M0-N0-C:J08-U11 "
            "2005-07-08-16.48.37.817558 R22-M0-N0-C:J08-U11 "
            "RAS KERNEL FAILURE link failed"
        )
        record = bgl.parse_line(line, 8)

        self.assertIsNotNone(record)
        self.assertEqual(record["is_anomaly"], 1)
        self.assertEqual(record["level"], "FAILURE")
        self.assertEqual(record["content"], "link failed")


class OpenSSHParserTests(unittest.TestCase):
    def setUp(self):
        openssh.reset_parser_state()

    def test_parses_invalid_user_and_source_ip(self):
        line = "Dec 10 07:07:38 LabSZ sshd[24206]: Invalid user test9 from 52.80.34.196"
        record = openssh.parse_line(line, 10)

        self.assertIsNotNone(record)
        self.assertEqual(record["line_id"], 10)
        self.assertEqual(record["level"], "INVALID_USER")
        self.assertEqual(record["node"], "LabSZ")
        self.assertEqual(record["source_ip"], "52.80.34.196")
        self.assertEqual(record["user"], "test9")
        self.assertGreater(record["timestamp"], 0)

    def test_unrecognized_month_is_rejected(self):
        line = "Foo 10 07:07:38 LabSZ sshd[24206]: Invalid user test9 from 52.80.34.196"
        record = openssh.parse_line(line, 11)

        self.assertIsNotNone(record)
        self.assertIsNone(record["timestamp"])

    def test_non_sshd_line_is_skipped(self):
        record = openssh.parse_line("this is not an sshd log line", 12)
        self.assertIsNone(record)


if __name__ == "__main__":
    unittest.main()
