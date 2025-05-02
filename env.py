import numpy as np
class CliffWalkingEnv:
    def __init__(self):
        self.height, self.width = 4, 12
        self.start_state = (3, 0)
        self.goal_state = (3, 11)
        self.cliff = [(3, c) for c in range(1, 11)]
        self.actions = {0:(-1,0),1:(0,1),2:(1,0),3:(0,-1)}
        self.reset()
    def _idx(self,pos): return pos[0]*self.width+pos[1]
    def state_from_idx(self,i): return divmod(i,self.width)
    def reset(self):
        self.agent_pos = self.start_state
        return self._idx(self.agent_pos)
    def step(self,a):
        dr,dc = self.actions[a]
        r,c = self.agent_pos[0]+dr, self.agent_pos[1]+dc
        r = max(0,min(r,self.height-1))
        c = max(0,min(c,self.width-1))
        reward, done = -1, False
        if (r,c) in self.cliff:
            reward, done = -100, True
            r,c = self.start_state
        elif (r,c)==self.goal_state:
            done = True
        self.agent_pos = (r,c)
        return self._idx(self.agent_pos), reward, done
