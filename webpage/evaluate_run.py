import os, json, glob
import cPickle as pickle
import ROOT as r
r.gROOT.SetBatch(1)
r.gStyle.SetOptStat(0)
r.gErrorIgnoreLevel = r.kWarning
import numpy as np
from histDefs import histDefs
import utils

def evaluateRun(run, indir, outdir):
    fin = r.TFile(os.path.join(indir,"{0}.root".format(run)))
    hpath = "DQMData/Run {{}}/CSC/Run summary/CSCOfflineMonitor/{0}/{1}"
    outdirr = os.path.join(outdir, str(run))
    os.system("rm -rf " + outdirr)
    os.system("mkdir -p " + outdirr)
    os.system("cp ~/scripts/index.php " + outdirr)
    outdirj = os.path.join(outdir, "jsons")
    os.system("mkdir -p " + outdirj)
    ntot = 0
    nflagged = 0
    score = 0
    for dname, hname in histDefs:
        _, good_rows, _, _ = pickle.load(open("hdata_pickles/{0}_{1}.pkl".format(dname, hname), 'rb'))
        nbins = good_rows.size
        d = pickle.load(open("../tf_test/out_pickles/{0}_{1}.pkl".format(dname, hname), 'rb'))
        hdata, _, _, n_ent = utils.readHistsFromFiles(indir, hpath.format(dname,hname), 10000, [run], good_rows)
        if hdata.size==0:
            continue

        res = utils.evaluateAE(hdata, d["weights"], d["biases"])

        sse = np.sqrt(np.sum((res-hdata)**2))
        pct = 1.0*np.sum(d["sse_array"] > sse) / d["sse_array"].size
        
        score += np.log(max(pct,0.001))

        ntot += 1
        if sse > d["thresh_5pct"]:
            nflagged += 1
            # print dname, hname
            # print hdata.shape, sse, pct

            ho = r.TH1D("ho_{0}_{1}".format(dname, hname), "", nbins, 0, nbins)
            hr = r.TH1D("hr_{0}_{1}".format(dname, hname), "", nbins, 0, nbins)
            
            for ib in range(nbins):
                ho.SetBinContent(ib+1, hdata[0,ib])
                hr.SetBinContent(ib+1, res[0,ib])
            
            c = r.TCanvas()
            
            ho.SetLineColor(r.kBlue)
            ho.SetLineWidth(4)
            hr.SetLineColor(r.kRed)
            hr.SetLineWidth(4)
            hr.SetLineStyle(9)
            
            ho.Draw("HIST")
            hr.Draw("SAME HIST")
            
            text = r.TLatex()
            text.SetTextFont(62)
            text.SetTextColor(r.kBlue)
            text.SetTextSize(0.06)
            text.DrawLatexNDC(0.7, 0.60, "{0:.3f}".format(sse))
            text.DrawLatexNDC(0.7, 0.55, "{0:.3f}".format(pct))
            text.DrawLatexNDC(0.7, 0.50, "{0:d}".format(int(n_ent[0])))
            
            c.SaveAs("{0}/{1}_{2}.png".format(outdirr,dname,hname))

    score = np.exp(score/ntot) if ntot>10 else 1.0
    rd = {"ntot":ntot, "nflagged":nflagged, "score":score}
    json.dump(rd, open(os.path.join(outdirj,"{0}.json".format(run)), 'w'))

def makeRunDirectoryPage(outdir, jsondir, run_info_file):
    run_info = json.load(open(run_info_file))
    runs = [int(x.split("/")[-1].split(".")[0]) for x in glob.glob(os.path.join(jsondir, "*.json"))]
    runs = sorted(runs, reverse=True)
    fout = open(os.path.join(outdir, "index.html"), 'w')

    fout.write("""<html><body>""")
    fout.write(open("style.css").read())

    fout.write("""
<h2>Runs</h2>
<table id="tg-2wHOX" class="tg">
    <th class="tg-baqh"><u>Run</u></th>
    <th class="tg-baqh"><u>Lumi</u></th>
    <th class="tg-baqh"><u>N Flagged</u></th>
    <th class="tg-baqh"><u>N Total</u></th>
    <th class="tg-baqh"><u>Score</u></th>
""")

    for i,run in enumerate(runs):
        if str(run) in run_info:
            ilumi = run_info[str(run)]["Initial Lumi"]
            elumi = run_info[str(run)]["Ending Lumi"]
            lumi = ilumi if ilumi==elumi else (ilumi-elumi)/np.log(ilumi/elumi)
        else:
            lumi = -1
        if i%2==0:
            cid = "yw4l"
        else:
            cid = "6k2t"
        info = json.load(open(os.path.join(jsondir, "{0}.json".format(run))))
        fout.write("""
<tr>
  <td class="tg-{0}"><a href="{1}">{1}</a></td>
  <td class="tg-{0}">{2:.1f}</td>
  <td class="tg-{0}">{3:d}</td>
  <td class="tg-{0}">{4:d}</td>
  <td class="tg-{0}">{5:.4f}</td>
</tr>
""".format(cid, run, lumi, info["nflagged"], info["ntot"], info["score"]))

    fout.write("</table><br><br>\n")
    fout.write("</body></html>\n")

    fout.close()

        

if __name__=="__main__":

    for f in glob.glob("/nfs-6/userdata/bemarsh/CSC_DQM/Run2018/SingleMuon/*.root"):
        run = f.split("/")[-1].split(".")[0]
        outdir = "/home/users/bemarsh/public_html/dump/DQMML_test/runs"
        # if os.path.exists(os.path.join(outdir, str(run))):
        #     continue
        print run
        evaluateRun(run, 
                    "/nfs-6/userdata/bemarsh/CSC_DQM/Run2018/SingleMuon/", 
                    "/home/users/bemarsh/public_html/dump/DQMML_test/runs")

    makeRunDirectoryPage("/home/users/bemarsh/public_html/dump/DQMML_test/runs",
                         "/home/users/bemarsh/public_html/dump/DQMML_test/runs/jsons",
                         "../run_info.json")

