"""HW6 Part C: explicit finite Grid World MDP, value iteration and rollouts."""
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / 'outputs'
OUTPUT.mkdir(exist_ok=True)
MAPS = {'original': ('.M.G', '.#IH', 'S...'), 'extended': ('.M.G', '.#IH', 'SD.W')}
# Floor rewards and movement models are data, independent of value iteration.
FLOORS = {'.': -1, 'S': -1, 'M': -3, 'I': -1, 'D': -2, 'W': -5, 'G': 10, 'H': -10}
DIRECTIONS = {'U': (-1, 0), 'R': (0, 1), 'D': (1, 0), 'L': (0, -1)}
ORDER = tuple(DIRECTIONS) # ties: U > R > D > L
PERPENDICULAR = {'U': ('L','R'), 'R': ('U','D'), 'D': ('L','R'), 'L': ('U','D')}
ARROWS = {'U':'↑','R':'→','D':'↓','L':'←'}
GAMMA = 0.9

class GridWorld:
    def __init__(self, lines):
        self.grid = tuple(lines)
        assert len(set(map(len, self.grid))) == 1
        assert all(t in FLOORS or t == '#' for line in self.grid for t in line)
        self.nrows, self.ncols = len(lines), len(lines[0])
        self.states = [(r,c) for r in range(self.nrows) for c in range(self.ncols) if self.grid[r][c] != '#']
        self.start = next(s for s in self.states if self.tile(s) == 'S')
    def tile(self, s):
        return self.grid[s[0]][s[1]]
    def terminal(self,s):
        return self.tile(s) in ('G','H')
    def move(self,s, direction):
        dr,dc = DIRECTIONS[direction]
        nr,nc = s[0]+dr,s[1]+dc
        if nr < 0 or nr >= self.nrows or nc < 0 or nc >= self.ncols or self.grid[nr][nc] == '#':
            return s
        return nr,nc
    def transitions(self,s,action):
        """List of (probability, destination, reward, terminated); duplicate destinations valid."""
        if self.terminal(s):
            return [(1.0,s,0,True)]
        moves = [(action,1.0)] if self.tile(s) != 'I' else [(action,.8), (PERPENDICULAR[action][0],.1), (PERPENDICULAR[action][1],.1)]
        ans = []
        for direction,prob in moves:
            dst = self.move(s,direction)
            ans.append((prob,dst,FLOORS[self.tile(dst)],self.terminal(dst)))
        assert abs(sum(p for p,_,_,_ in ans)-1) < 1e-12
        return ans
    def q(self,s,a,values):
        return sum(p * (rew + GAMMA*values[dst]) for p,dst,rew,_ in self.transitions(s,a))
    def update(self,values):
        return {s:(0.0 if self.terminal(s) else max(self.q(s,a,values) for a in ORDER)) for s in self.states}
    def best_actions(self,s,values,tol=1e-9):
        if self.terminal(s): return []
        qs = {a:self.q(s,a,values) for a in ORDER}
        best = max(qs.values())
        return [a for a in ORDER if abs(qs[a] - best) <= tol]
    def solve(self,tolerance=1e-6,limit=10000):
        values = {s:0.0 for s in self.states}
        v1 = self.update(values)
        v2 = self.update(v1)
        change = float('inf')
        for iteration in range(1,limit+1):
            next_values = self.update(values)
            change = max(abs(next_values[s] - values[s]) for s in self.states)
            values = next_values
            if change < tolerance:
                return values,iteration,change,v1,v2,True
        return values,limit,change,v1,v2,False
    def value_map(self,values):
        return '\n'.join(' '.join('#####' if self.grid[r][c]=='#' else f'{values[(r,c)]:7.3f}' for c in range(self.ncols)) for r in range(self.nrows))
    def policy_map(self,values):
        return '\n'.join('  '.join(self.grid[r][c] if self.grid[r][c] in ('G','H','#') else ''.join(ARROWS[a] for a in self.best_actions((r,c),values)) for c in range(self.ncols)) for r in range(self.nrows))
    def sample_step(self,s,a,rng):
        if self.terminal(s): return s,0,True
        x = rng.random()
        cum = 0
        for p,dst,reward,done in self.transitions(s,a):
            cum += p
            if x < cum + 1e-12: return dst,reward,done
        raise AssertionError('invalid probabilities')

def checks():
    g = GridWorld(MAPS['original']); h = GridWorld(MAPS['extended']); lines=[]
    def record(name,condition,detail):
        assert condition,name
        lines.append(f'PASS {name}: {detail}')
    record('boundary stays and gets floor reward', g.transitions((0,0),'U') == [(1.0,(0,0),-1,False)],str(g.transitions((0,0),'U')))
    record('wall stays and gets mud reward',g.transitions((0,1),'D') == [(1.0,(0,1),-3,False)],str(g.transitions((0,1),'D')))
    for a in ORDER:
        t = g.transitions((1,2),a)
        record(f'ice probabilities {a}',len(t)==3 and sorted(p for p,_,_,_ in t)==[.1,.1,.8],str(t))
    record('mud entry reward',g.transitions((0,0),'R') == [(1.0,(0,1),-3,False)],str(g.transitions((0,0),'R')))
    record('sand entry reward',h.transitions((2,0),'R') == [(1.0,(2,1),-2,False)],str(h.transitions((2,0),'R')))
    record('water entry reward',h.transitions((2,2),'R') == [(1.0,(2,3),-5,False)],str(h.transitions((2,2),'R')))
    record('goal entry +10 no extra -1',g.transitions((0,2),'R') == [(1.0,(0,3),10,True)],str(g.transitions((0,2),'R')))
    record('hole entry -10',g.transitions((2,3),'U') == [(1.0,(1,3),-10,True)],str(g.transitions((2,3),'U')))
    record('goal terminal no further reward',g.transitions((0,3),'L') == [(1.0,(0,3),0,True)],str(g.transitions((0,3),'L')))
    record('hole terminal no further reward',g.transitions((1,3),'L') == [(1.0,(1,3),0,True)],str(g.transitions((1,3),'L')))
    return '\n'.join(lines)+'\n'

def rollout(g,values,seed,limit=100):
    rng = random.Random(seed)
    s = g.start
    ret = 0.0
    trace=[]
    for step in range(1,limit+1):
        a = g.best_actions(s,values)[0]
        dst,rew,done = g.sample_step(s,a,rng)
        ret += GAMMA**(step-1)*rew
        trace.append(f'{step:03d}: {s} {g.tile(s)} --{a}--> {dst} {g.tile(dst)} reward={rew}')
        s = dst
        if done:
            return dict(steps=step,discounted_return=ret,outcome=('goal' if g.tile(s)=='G' else 'hole')),trace
    return dict(steps=limit,discounted_return=ret,outcome='time_limit'),trace

def main():
    (OUTPUT/'rule_checks.txt').write_text(checks(),encoding='utf8')
    allrows=[]; summaries=[]; report=[]
    for name,lines in MAPS.items():
        g=GridWorld(lines)
        values,n,delta,v1,v2,converged=g.solve()
        assert converged, f'NONCONVERGENCE {name} after {n} iterations'
        report.extend([f'MAP {name}', '\n'.join(lines),f'V1:\n{g.value_map(v1)}',f'V2 (1,2)={v2[(0,1)]:.6f}, (1,3)={v2[(0,2)]:.6f}, (2,3)={v2[(1,2)]:.6f}',f'Iterations={n}, last max change={delta:.12g}, converged={converged}',f'Final values:\n{g.value_map(values)}',f'Final policy (ties in U,R,D,L order):\n{g.policy_map(values)}',''])
        (OUTPUT/f'{name}_values_policy.txt').write_text('\n'.join(report[-7:]),encoding='utf8')
        traces={}
        for batch in (11,22,33):
            rows=[]
            for i in range(50):
                seed=1000*batch+i
                res,trace=rollout(g,values,seed)
                row=dict(map=name,batch_seed=batch,episode_seed=seed,**res)
                rows.append(row); allrows.append(row)
                traces.setdefault(res['outcome'],(seed,trace))
            summary=dict(map=name,batch_seed=batch,goals=sum(r['outcome']=='goal' for r in rows),holes=sum(r['outcome']=='hole' for r in rows),time_limits=sum(r['outcome']=='time_limit' for r in rows),mean_steps=sum(r['steps'] for r in rows)/50,mean_discounted_return=sum(r['discounted_return'] for r in rows)/50)
            summaries.append(summary)
        for outcome,(seed,trace) in traces.items():
            (OUTPUT/f'{name}_{outcome}_trace.txt').write_text(f'Map={name} outcome={outcome} episode_seed={seed}\n'+'\n'.join(trace)+'\n',encoding='utf8')
    for file,rows,fields in [(OUTPUT/'part_c_episodes.csv',allrows,['map','batch_seed','episode_seed','steps','discounted_return','outcome']), (OUTPUT/'part_c_summary.csv',summaries,['map','batch_seed','goals','holes','time_limits','mean_steps','mean_discounted_return'])]:
        with file.open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (OUTPUT/'part_c_solver_report.txt').write_text('\n'.join(report),encoding='utf8')
    print('\n'.join(report))
    print('PART C BATCH SUMMARIES:')
    for row in summaries: print(row)

if __name__=='__main__': main()
