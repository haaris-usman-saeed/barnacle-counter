import json
import unittest

import numpy as np

from tests.helpers import ROOT
from barnacles.review import replay, model_set, neighbours_from_pairs


def toy():
    # five candidates; 0 and 1 overlap (pair), 4 is low-scoring
    c = {"id": [f"c{i}" for i in range(5)], "x": [10, 12, 50, 80, 120], "y": [10, 10, 50, 80, 120],
         "u": [0.05, 0.06, 0.25, 0.4, 0.95], "v": [0.05, 0.05, 0.25, 0.4, 0.95],
         "ml_score": [0.9, 0.8, 0.7, 0.6, 0.1]}
    return c, neighbours_from_pairs(5, [(0, 1)])


def session(actions, thr=0.5):
    return {"operating_point": {"threshold": thr, "initial_threshold": thr}, "grid": {"rows": 2, "cols": 2},
            "actions": actions}


class TestCorrections(unittest.TestCase):
    def test_model_set(self):
        c, nb = toy()
        self.assertEqual(model_set(c["ml_score"], 0.5, nb), [0, 2, 3])

    def test_remove_keep_add_restore(self):
        c, nb = toy()
        r = replay(c, nb, session([]))
        self.assertEqual((r["model_count"], r["corrected_count"]), (3, 3))
        r = replay(c, nb, session([{"action": "remove", "id": "c2"}]))
        self.assertEqual(r["corrected_count"], 2)
        r = replay(c, nb, session([{"action": "remove", "id": "c2"}, {"action": "restore", "id": "c2"}]))
        self.assertEqual(r["corrected_count"], 3)
        r = replay(c, nb, session([{"action": "keep", "id": "c4"}]))
        self.assertEqual(r["corrected_count"], 4)
        self.assertIn("reviewer_accepted_suggestion", {d["source"] for d in r["detections"]})
        add = {"action": "add", "id": "a1", "x": 70, "y": 20, "u": 0.7, "v": 0.2}
        r = replay(c, nb, session([add]))
        self.assertEqual(r["corrected_count"], 4)
        r = replay(c, nb, session([add, {"action": "remove", "id": "a1"}]))
        self.assertEqual(r["corrected_count"], 3)

    def test_threshold_action_and_overrides_persist(self):
        c, nb = toy()
        acts = [{"action": "remove", "id": "c0"}, {"action": "set_threshold", "threshold": 0.05}]
        r = replay(c, nb, session(acts))
        # lower threshold adds c4; c0 still wins its pair (c1 stays suppressed) but is removed
        self.assertEqual(r["model_count"], 4)
        self.assertEqual(r["corrected_count"], 3)

    def test_dashboard_breakdown(self):
        c, nb = toy()
        acts = [{"action": "keep", "id": "c0"}, {"action": "remove", "id": "c2"}, {"action": "keep", "id": "c4"},
                {"action": "remove", "id": "c1"},
                {"action": "add", "id": "a1", "x": 1, "y": 1, "u": 0.6, "v": 0.6}]
        r = replay(c, nb, session(acts))
        b = r["breakdown"]
        self.assertEqual(b, {"selected_model": 3, "confirmed": 1, "removed": 1, "accepted_suggestions": 1,
                             "rejected_suggestions": 1, "added": 1, "reviewed": 4})
        self.assertEqual(b["selected_model"] - b["removed"] + b["accepted_suggestions"] + b["added"], r["corrected_count"])

    def test_counts_reconcile(self):
        c, nb = toy()
        r = replay(c, nb, session([{"action": "keep", "id": "c4"},
                                   {"action": "add", "id": "a1", "x": 1, "y": 1, "u": 0.6, "v": 0.6}]))
        self.assertEqual(sum(r["per_cell_corrected"].values()), r["corrected_inside_frame"])
        self.assertEqual(r["corrected_inside_frame"], r["corrected_count"])

    def test_ui_session_replays_identically(self):
        """A session exported by the browser interface (Playwright run against the app, after
        uploading a photo, accepting, removing, adding and undoing) replays to the same
        result in Python."""
        z = json.loads((ROOT / "tests" / "fixtures" / "ui_session_upload.json").read_text())
        s, d = z["session"], z["data"]
        nb = [[] for _ in d["c"]["s"]]
        for k in range(0, len(d["pairs"]), 2):
            nb[d["pairs"][k]].append(d["pairs"][k + 1])
            nb[d["pairs"][k + 1]].append(d["pairs"][k])
        cands = {"id": d["c"]["id"], "x": d["c"]["x"], "y": d["c"]["y"], "u": d["c"]["u"], "v": d["c"]["v"],
                 "ml_score": d["c"]["s"]}
        r = replay(cands, nb, s)
        self.assertEqual(r["corrected_count"], s["result"]["corrected_count"])
        self.assertEqual([(q["id"], q["source"]) for q in r["detections"]],
                         [(q["id"], q["source"]) for q in s["result"]["detections"]])
        self.assertEqual(r["per_cell_corrected"], s["result"]["per_cell_corrected"])
        # the fixture was recorded after an undo: 4 actions remain of 5
        self.assertEqual(len(s["actions"]), 4)

if __name__ == "__main__":
    unittest.main()
