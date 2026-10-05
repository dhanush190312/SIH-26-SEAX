import json
import threading
import unittest
import urllib.error
import urllib.request
from dataclasses import replace
from common import isolated_data_dir, make_cfg
from orca.api.server import create_server


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = replace(make_cfg(), data_dir=isolated_data_dir())
        cls.srv = create_server(cls.cfg, "127.0.0.1", 0)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def call(self, path, body=None):
        req = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if r.headers["Content-Type"].startswith("application/json") else raw)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_health_and_ui(self):
        s, h = self.call("/api/orca/health")
        self.assertEqual((s, h["status"]), (200, "ok"))
        self.assertTrue(h["modes"]["DEMO"]["capabilities"]["weather"])
        self.assertEqual(h["languages"], ["en", "ta"])
        s, html = self.call("/")
        self.assertEqual(s, 200)
        self.assertIn(b"WHY THIS RESULT?", html)
        self.assertNotIn(b"http://", html.replace(b"http://www.w3.org", b""), "the UI must not depend on any CDN")
        self.assertNotIn(b"https://", html)

    def test_query_then_trace_evidence_report(self):
        s, r = self.call("/api/orca/query", {"text": "Can I fish tomorrow at 6 AM from Nagapattinam?", "session_id": "api"})
        self.assertEqual((s, r["state"]), (200, "LOWER_RISK"))
        qid = r["query_id"]
        s, tr = self.call(f"/api/orca/trace/{qid}")
        self.assertEqual(s, 200)
        self.assertEqual(tr["query_id"], qid)
        s, ev = self.call(f"/api/orca/evidence/{qid}")
        self.assertGreater(ev["n"], 5)
        s, rep = self.call(f"/api/orca/report/{qid}")
        self.assertIn(b"ORCA report", rep)
        s, res = self.call(f"/api/orca/result/{qid}")
        self.assertNotIn("evidence", res)

    def test_bad_requests_get_400_not_500(self):
        for path, body in (("/api/orca/query", {}), ("/api/orca/query", {"text": "x" * 600}), ("/api/orca/weather", {"latitude": "abc", "longitude": 1}),
                           ("/api/orca/weather", {"latitude": 95, "longitude": 1, "start_time": "2026-10-01T06:00"}),
                           ("/api/orca/weather", {"latitude": 10, "longitude": 80, "start_time": "garbage"}),
                           ("/api/orca/compare", {"times": ["2026-10-01T06:00:00"]}),
                           ("/api/orca/query", {"text": "hi", "mode": "NOPE"})):
            s, e = self.call(path, body)
            self.assertEqual(s, 400, (path, body, e))
            self.assertIn("error", e)

    def test_unknown_routes_and_ids(self):
        self.assertEqual(self.call("/nope")[0], 404)
        self.assertEqual(self.call("/api/orca/trace/o_00000000")[0], 404)
        self.assertEqual(self.call("/api/orca/trace/..%2f..%2fetc%2fpasswd")[0], 400)

    def test_direct_tool_endpoints_share_the_same_evidence(self):
        s, w = self.call("/api/orca/weather", {"latitude": 10.765, "longitude": 79.87, "start_time": "2026-10-01T10:00:00"})
        self.assertEqual(s, 200)
        conflicts = [c for sl in w["slots"] for c in sl["conflicts"]]
        self.assertEqual([c["variable"] for c in conflicts], ["wave_height"])
        s, o = self.call("/api/orca/ocean", {"latitude": 11.0, "longitude": 80.12, "start_time": "2026-10-01T10:00:00"})
        self.assertIn("chlorophyll", {x["variable"] for x in o["observations"]})
        s, g = self.call("/api/orca/geo", {"latitude": 10.74, "longitude": 80.14, "date": "2026-10-01"})
        self.assertEqual(g["geofence_status"]["inside"][0]["id"], "GF-NAVAL")
        self.assertTrue(g["pfz"]["available"])

    def test_route_and_compare_endpoints(self):
        s, r = self.call("/api/orca/route", {"place": "Nagapattinam", "time": "2026-10-01T10:00:00"})
        self.assertEqual(s, 200)
        self.assertTrue(any(x.get("feasible") for x in r["routes"]["routes"]))
        s, c = self.call("/api/orca/compare", {"place": "Nagapattinam", "times": ["2026-10-01T06:00:00", "2026-10-01T10:00:00"]})
        self.assertEqual([x["state"] for x in c["comparison"]], ["LOWER_RISK", "MODERATE_CONDITIONS"])

    def test_offline_flow_refresh_then_use(self):
        s, r = self.call("/api/orca/query", {"text": "Can I fish tomorrow at 6 AM from Nagapattinam?", "mode": "OFFLINE", "session_id": "off"})
        self.assertEqual(r["status"], "no_data")
        s, c = self.call("/api/orca/cache/refresh", {})
        self.assertTrue(c["ok"])
        s, r = self.call("/api/orca/query", {"text": "Can I fish tomorrow at 6 AM from Nagapattinam?", "mode": "OFFLINE", "session_id": "off2"})
        self.assertEqual(r["state"], "LOWER_RISK")
        self.assertIn("CACHED DEMO DATA", r["mode_label"])
        self.assertIn("last updated", r["mode_label"])

    def test_scenarios_endpoint(self):
        s, sc = self.call("/api/orca/scenarios")
        self.assertEqual(len(sc["scenarios"]), 11)
