import sys
import experiments.nav_compare.scripts.compare_aiding_outage as m
m.CASES = {k: v for k, v in m.CASES.items() if k in ("ugv_short", "ugv_static_short")}
m.FILTERS = {k: v for k, v in m.FILTERS.items() if k in ("ESKF", "InEKF")}
sys.argv = ["x", "--seeds", "5", "--out", sys.argv[1], "--workers", "2"]
m.main()
