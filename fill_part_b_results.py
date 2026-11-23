"""Insert actual Part B results into SUBMISSION.md after frozenlake.py runs."""
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parent
CSV = ROOT/'outputs'/'part_b_summary.csv'
SUB = ROOT/'SUBMISSION.md'
START = '<!-- PART_B_RESULTS_START -->'
END = '<!-- PART_B_RESULTS_END -->'

def main():
    rows=list(csv.DictReader(CSV.open(encoding='utf8')))
    assert len(rows)==12, f'Expected 12 rows, got {len(rows)}'
    lines=['| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['slippery']} | {r['policy']} | {r['batch_seed']} | {r['goals']} | {r['holes']} | {r['time_limits']} | {float(r['success_percent']):.1f}% | {float(r['mean_steps']):.2f} |")
    content=SUB.read_text(encoding='utf8')
    assert START in content and END in content
    left,rest=content.split(START,1)
    _,right=rest.split(END,1)
    SUB.write_text(left+START+'\n'+'\n'.join(lines)+'\n'+END+right,encoding='utf8')
    print('Inserted real Part B table in SUBMISSION.md')

if __name__=='__main__':main()
