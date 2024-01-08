import numpy as np
class MDP:
    def __init__(self, num_states, num_actions, end_states, Transition_prob,rewards, mdptype, discount):
        self.num_states = num_states
        self.num_actions = num_actions
        self.end_states = end_states
        self.Transition_prob=Transition_prob
        self.rewards = rewards
        self.mdptype = mdptype
        self.discount = discount
      

        if mdptype=="continuing":
            self.continuing_mdp_solving()
        else:
            self.episodic_mdp_solving()
    
    def  continuing_mdp_solving(self):
        
        V=np.zeros(self.num_states)
        epsilon =1e-6
        pi = np.zeros(num_states, dtype=int)

        for i in range(1000):
             delta=0
             V1=V.copy()
             pi1=pi.copy()
             for s in range(self.num_states):
                actions=[]
                for a in range(self.num_actions):    
                    sum=0
                    for _s in range(self.num_states):
                        if Transition_prob[s][a][_s]!=0:
                            sum+=Transition_prob[s][a][_s]*(self.rewards[s][a][_s] + self.discount * V[_s])    
                    actions.append(sum)
                V1[s]=max(actions)
                pi1[s]=np.argmax(actions)

                delta = max(delta, abs(V[s]- V1[s]))
             V=V1
             pi=pi1
             if delta<epsilon:
                break
                  
        output_file = f"sol-{self.mdptype}-mdp-{self.num_states}-{self.num_actions}.txt"
        with open(output_file, 'w') as outfile:
            for i in range(num_states):
                outfile.write(f"{np.round(V[i],6)} {pi[i]}\n")
   
    def episodic_mdp_solving(self):
        V=np.zeros(self.num_states)
        epsilon =1e-6
        pi = np.zeros(num_states, dtype=int)

        for i in range(1000):
             delta=0
             V1=V.copy()
             pi1=pi.copy()
             for s in range(self.num_states):
                
                actions=[]
                if s in self.end_states:
                    V1[s]=0
                    pi1[s]=0
                else:
                    for a in range(self.num_actions):    
                        sum=0

                        for _s in range(self.num_states):
                            if Transition_prob[s][a][_s]!=0:
                                sum+=Transition_prob[s][a][_s]*(self.rewards[s][a][_s] + self.discount * V[_s])    
                        actions.append(sum)
                    V1[s]=max(actions)
                    pi1[s]=np.argmax(actions)

                    delta = max(delta, abs(V[s]- V1[s]))
             if delta<epsilon:
                        break       
             V=V1
             pi=pi1
             
        output_file = f"sol-{self.mdptype}-mdp-{self.num_states}-{self.num_actions}.txt"
        with open(output_file, 'w') as outfile:
            for i in range(num_states):
                outfile.write(f"{np.round(V[i],6)} {pi[i]}\n")

if __name__ == "__main__":
    input_files = ["continuing-mdp-2-2.txt","continuing-mdp-10-5.txt","continuing-mdp-50-20.txt","episodic-mdp-2-2.txt","episodic-mdp-10-5.txt","episodic-mdp-50-20.txt"] # Replace with your input file
    for input_file in input_files:
        with open(input_file, 'r') as file:
                lines = file.readlines()

            # Extract values from lines (you may need to adjust this based on your actual input file format)
        num_states = int(lines[0].split()[1])
        num_actions = int(lines[1].split()[1])
        end_states = lines[2].split()[1:]
        Transition_prob=np.zeros((num_states,num_actions,num_states))
        rewards=np.zeros((num_states,num_actions,num_states))
        for line in lines[3:-2]:
            Transition_prob[int(line.split()[1])][int(line.split()[2])][int(line.split()[3])]=float(line.split()[5])
            rewards[int(line.split()[1])][int(line.split()[2])][int(line.split()[3])]=float(line.split()[4]) # Adjust as needed
        
        mdptype = lines[-2].split()[1]
        discount = float(lines[-1].split()[1])

        mdp = MDP(num_states, num_actions, end_states, Transition_prob,rewards, mdptype, discount)

    
