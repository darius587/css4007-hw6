"""HW6 Part B: reproducible Gymnasium FrozenLake experiments."""
import csv
import random
import sys
from pathlib import Path

import gymnasium as gym

OUT = Path(__file__).resolve().parent / 'outputs'
OUT.mkdir(exist_ok=True)
MAP = ('SFFF','FHFH','FFFH','HFFG')
# Gymnasium actions: 0 left, 1 down, 2 right, 3 up.
# This route from S follows top row and avoids hole at (1,3):
# (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (3,2) -> (3,3).
# Fixed policy defined for every nonterminal index, even if unreachable on this route.
FIXED = {0:2,1:2,2:1,3:0,4:3,6:1,8:2,9:2,10:1,13:2,14:2}
NONTERMINAL = {r*4+c for r,line in enumerate(MAP) for c,t in enumerate(line) if t not in 'HG'}
assert set(FIXED)==NONTERMINAL


def episode(env, policy, episode_seed, action_rng, render=False):
    observation, _ = env.reset(seed=episode_seed)
    path = [f'reset(seed={episode_seed}), state={observation}\n{env.render()}' if render else '']
    total=0.0
    for step in range(1,101):
        action = action_rng.randrange(4) if policy=='random' else FIXED[int(observation)]
        observation, reward, terminated, truncated, _ = env.step(action)
        total += reward
        if render:
            path.append(f'step={step} action={action} state={observation} reward={reward} terminated={terminated} truncated={truncated}\n{env.render()}')
        if terminated or truncated or step==100:
            # Check terminal state first, including if terminal happened on step 100.
            final_tile=MAP[int(observation)//4][int(observation)%4]
            outcome='goal' if terminated and final_tile=='G' else ('hole' if terminated and final_tile=='H' else 'time_limit')
            return dict(steps=step,total_reward=total,outcome=outcome), '\n'.join(path)
    raise AssertionError('Episode did not stop')


def main():
    rows=[]; summary=[]; traces={}
    for slippery in (False,True):
        env=gym.make('FrozenLake-v1',map_name='4x4',is_slippery=slippery,render_mode='ansi',max_episode_steps=100)
        for policy in ('random','fixed'):
            for batch in (11,22,33):
                batch_rows=[]
                for i in range(50):
                    seed=1000*batch+i
                    # Random action generator is seeded independently for each episode.
                    # Reusing the episode seed ensures comparable conditions.
                    action_rng=random.Random(seed)
                    result,trace=episode(env,policy,seed,action_rng,render=True)
                    row=dict(condition=f'slippery={slippery}/{policy}',slippery=slippery,policy=policy,batch_seed=batch,episode_seed=seed,**result)
                    rows.append(row);batch_rows.append(row)
                    traces.setdefault(result['outcome'],(row,trace))
                goals=sum(r['outcome']=='goal' for r in batch_rows)
                holes=sum(r['outcome']=='hole' for r in batch_rows)
                time_limits=sum(r['outcome']=='time_limit' for r in batch_rows)
                summary.append(dict(slippery=slippery,policy=policy,batch_seed=batch,goals=goals,holes=holes,time_limits=time_limits,success_percent=100*goals/50,mean_steps=sum(r['steps'] for r in batch_rows)/50))
        env.close()
    for name,table,fields in [
        ('part_b_episodes.csv',rows,['condition','slippery','policy','batch_seed','episode_seed','steps','total_reward','outcome']),
        ('part_b_summary.csv',summary,['slippery','policy','batch_seed','goals','holes','time_limits','success_percent','mean_steps'])]:
        with (OUT/name).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(table)
    for outcome,(row,trace) in traces.items():
        (OUT/f'part_b_{outcome}_trace.txt').write_text(f'{row}\n{trace}\n',encoding='utf8')
    with (OUT/'part_b_versions.txt').open('w',encoding='utf8') as f:
        f.write(f'python: {sys.version}\ngymnasium: {gym.__version__}\n')
        try:
            import numpy
            f.write(f'numpy: {numpy.__version__}\n')
        except ImportError: pass
    from fill_part_b_results import main as fill_part_b_results
    fill_part_b_results()
    print('Gymnasium',gym.__version__)
    for record in summary: print(record)
    print('Saved',len(rows),'episodes and',len(summary),'batch summaries in outputs/')

if __name__=='__main__':main()
