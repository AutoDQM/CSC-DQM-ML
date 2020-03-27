from histDefs import histDefs
import utils

for dname, hname in histDefs:
    print dname, hname
    utils.load_hist_data(dname, hname, lumi_json="../../run_info.json", force_reload=True)
