import numpy as np

class state():
    def __init__(self, ball=None, target=None, M=7, init=True):
        self.ball = ball if ball is not None else np.random.randint(0,M,size=2) - (M-1)/2
        self.target=target if target is not None else np.random.randint(0,M,size=2) - (M-1)/2
        if init:
            while all(self.target == self.ball):
                self.target = np.random.randint(0,M,size=2) - (M-1)/2
        self.M=M
        

    def action(self,n):
        '''
        apply an action n to change the ball position, unless the ball falls outside the field.
        '''
        actions = np.array([(0,1),(1,1),(1,0), (1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)])
        if all((abs(self.ball + actions[n]) <= self.M / 2)):
            self.ball =self.ball + actions[n]
            
    def copy(self):
        s=state()
        s.ball = self.ball
        s.target = self.target
        s.M = self.M
        return s
    
    def symetries(self):
        '''
        compute three symmetric states, obtained by central symmetry, or axial symetries 
        (vertical and horizontal) 
        '''
        s_o = state(ball=-self.ball, target=-self.target, M=self.M, init=False)
        s_ox = state(ball = np.array((-self.ball[0],self.ball[1])),
                     target = np.array((-self.target[0],self.target[1])), 
                     M=self.M, init=False)
        s_oy = state(ball = -s_ox.ball,
                     target = -s_ox.target,
                     M = self.M, init = False)
        return s_o, s_ox, s_oy

     
        
class game():
    def __init__(self,M=7, t=0,r=[], end=False, s=None):
        self.history=[state(M=M)] if s is None else [s]
        self.t=t
        self.target = self.history[0].target
        self.reward=r
        self.end=end
        self.compute_Tmin()
    
    def compute_Tmin(self):
        self.Tmin = int(max(abs(self.history[0].ball[0]-self.history[0].target[0]),
                        abs(self.history[0].ball[1]-self.history[0].target[1])))
        
    def next(self,a):
        '''
        apply an action a to the current state
        '''
        if self.end ==False :
            self.t +=1
            self.history.append(self.history[-1].copy())
            self.history[-1].action(a)

            if all(self.history[-1].ball == self.target) :
                self.reward.append(1)
                self.end=True
            elif self.t==self.history[0].M*3:
                self.end=True
                self.reward.append(0)
            else:
                self.reward.append(0)
                
    def symetries(self):
        '''
        compute the symetric games, in order to speed up Q update
        ''' 
        k = 3 # number of symetries
        games= ()
        for i in range(k):
            games +=( game(M=self.history[0].M, t=self.t, r=self.reward, end=self.end) ,)
            games[i].history=[]
        
        for s in self.history:
            states = s.symetries()
            for i,episode in enumerate(games):
                episode.history.append(states[i])
                
        for episode in games:
            episode.compute_Tmin()
            episode.target = episode.history[-1].target
            
        return games
        
        
    def recover_actions(self):
        '''
        recover actions from a game history.
        This method is used to compute the symetric actions from each symetric game.
        For any action that led to keep the ball at the same position, the proposed action is
        randomly chosen among the actions leading to a field exit.
        '''
        actions = []
        actions_array= np.array([(0,1),(1,1),(1,0), (1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)])
        for k in range(len(self.history)-1):
            action = self.history[k+1].ball-self.history[k].ball
            if all(action == np.zeros(2)):
                action = actions_array[np.random.randint(0,8)]
                while all(abs(self.history[k].ball + action) <= self.history[k].M /2):
                    action = actions_array[np.random.randint(0,8)]                    
            actions.append( actions_array.tolist().index(action.tolist()) )
        return actions
            







