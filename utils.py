import numpy as np, matplotlib.pyplot as plt
directions = np.array(list('URDL'))
def policy_from_Q(Q): return directions[np.argmax(Q,axis=1)]
def plot_policy(env,Q,title,path):
    pol = policy_from_Q(Q).reshape(env.height,env.width)
    fig,ax = plt.subplots(figsize=(12,4))
    ax.set_title(title); ax.set_xticks(range(env.width)); ax.set_yticks(range(env.height))
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.grid(True)
    arr={'U':(0,-0.3),'D':(0,0.3),'L':(-0.3,0),'R':(0.3,0)}
    for r in range(env.height):
        for c in range(env.width):
            if (r,c) in env.cliff: ax.text(c,r,'X',ha='center',va='center',color='red',fontsize=14)
            elif (r,c)==env.start_state: ax.text(c,r,'S',ha='center',va='center',color='green',fontsize=14)
            elif (r,c)==env.goal_state: ax.text(c,r,'G',ha='center',va='center',color='blue',fontsize=14)
            else:
                dx,dy = arr[pol[r,c]]; ax.arrow(c,r,dx,dy,head_width=0.2,head_length=0.2,fc='k',ec='k')
    ax.set_xlim(-0.5,env.width-0.5); ax.set_ylim(env.height-0.5,-0.5); fig.tight_layout(); fig.savefig(path)
def plot_returns(data,path):
    plt.figure()
    for label,ret in data.items(): plt.plot(ret,label=label)
    plt.xlabel('Episode'); plt.ylabel('Return'); plt.legend(); plt.savefig(path,bbox_inches='tight')
