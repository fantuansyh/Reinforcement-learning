import numpy as np, random
def eg(Q,s,eps):
    return random.randrange(Q.shape[1]) if random.random()<eps else int(np.argmax(Q[s]))
def sarsa(env, episodes=500, alpha=0.5, gamma=1.0, eps=0.1):
    nS, nA = env.height*env.width, 4
    Q = np.zeros((nS,nA)); returns=[]
    for _ in range(episodes):
        s=env.reset(); a=eg(Q,s,eps); G=0; done=False
        while not done:
            s2,r,done = env.step(a)
            a2 = eg(Q,s2,eps)
            Q[s,a] += alpha*(r + gamma*Q[s2,a2] - Q[s,a])
            s,a = s2,a2; G+=r
        returns.append(G)
    return Q, returns
