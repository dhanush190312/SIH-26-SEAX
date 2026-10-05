import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from common import ROOT, make_cfg

UI = ROOT / "orca" / "api" / "ui.html"


@unittest.skipUnless(shutil.which("node"), "node not installed")
class UiTests(unittest.TestCase):
    """The browser code is checked without a browser: syntax, and the map renderer as a pure function."""

    @classmethod
    def setUpClass(cls):
        src = UI.read_text(encoding="utf-8")
        cls.js = re.search(r"<script>(.*)</script>", src, re.S).group(1)

    def run_node(self, code: str) -> str:
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "t.js"
            f.write_text(code, encoding="utf-8")
            r = subprocess.run(["node", str(f)], capture_output=True, text=True, timeout=30)
            self.assertEqual(r.returncode, 0, r.stderr)
            return r.stdout

    def test_js_parses(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "ui.js"
            f.write_text(self.js, encoding="utf-8")
            r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)

    def test_map_renderer_draws_real_layers(self):
        from orca.pipeline import Orca
        import json
        res = Orca(make_cfg()).query("Give me two route options avoiding restricted areas from Nagapattinam tomorrow at 10 AM.")
        fn = re.search(r"(const STATE_COL.*?)function drawMap", self.js, re.S).group(1)
        code = ('const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",\'"\':"&quot;"}[c]));\n' + fn +
                f"\nconst m={json.dumps(res['map'])};\nconst out=renderMapSVG(m,{{}});\nconsole.log(JSON.stringify({{polys:(out.match(/<polygon/g)||[]).length,circles:(out.match(/<circle/g)||[]).length,lines:(out.match(/<polyline/g)||[]).length,nan:/NaN|undefined/.test(out),hatch:out.includes('url(#hatch)'),empty:renderMapSVG(null,{{}}).includes('No map data'),off:renderMapSVG(m,{{routes:false}}).includes('<polyline')}}));")
        out = json.loads(self.run_node(code).strip())
        res2 = Orca(make_cfg()).query("Find the nearest PFZ from Nagapattinam and check the hazards around it tomorrow morning.")
        res3 = Orca(make_cfg()).query("Which nearby regions near Nagapattinam have high chlorophyll and favourable SST?")
        code2 = ('const esc=s=>String(s??"");\n' + fn + f"\nconst a=renderMapSVG({json.dumps(res2['map'])},{{}}), b=renderMapSVG({json.dumps(res3['map'])},{{}});console.log(JSON.stringify({{buffer:a.includes('stroke-dasharray=\"2 5\"'),regions:(b.match(/<polygon points=\"[0-9.,\\- ]+\" fill=\"#(1c7c4a|888)\"/g)||[]).length,nan:/NaN|undefined/.test(a+b)}}));")
        o2 = json.loads(self.run_node(code2).strip())
        self.assertTrue(o2["buffer"], "the hazard analysis area is drawn")
        self.assertGreaterEqual(o2["regions"], 4, "ocean regions are drawn")
        self.assertFalse(o2["nan"])
        self.assertFalse(out["nan"], "no NaN/undefined coordinates")
        self.assertGreaterEqual(out["polys"], 5)
        self.assertGreaterEqual(out["circles"], 10)
        self.assertGreaterEqual(out["lines"], 1)
        self.assertTrue(out["hatch"])
        self.assertTrue(out["empty"])
        self.assertFalse(out["off"], "layer toggles work")
