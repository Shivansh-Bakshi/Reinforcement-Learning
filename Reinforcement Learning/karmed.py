import numpy as np
import matplotlib.pyplot as plt
import ray

ray.init()

def createState(k):
    #Initialie q_star using normal distribution
    mu, sigma = 0, 1
    # Mean 0 Variance 1
    q_star = np.random.normal(mu, sigma, k)

    R = np.zeros(k)
    for i in range(k):
        mu_r = q_star[i]
        sigma_r = 1
        # Mean q_star(a) variance 1
        R[i] = np.random.normal(mu_r, sigma_r, 1)

    return q_star, R

def updateEstimate(action, prevEstimate, rewards, stepSize):
    return(prevEstimate[action] + stepSize*(rewards[action] - prevEstimate[action]))

def updateRewards(q_star, k, R):
    for i in range(k):
        mu_r = q_star[i]
        sigma_r = 1
        # Mean q_star(a) variance 1
        R[i] = np.random.normal(mu_r, sigma_r, 1)
    return R

def bandit(k, e, steps):
    # Q (estimate action-value) is initialized to 0
    Q = np.zeros(k)

    # We do not have access to this information as an agent
    q_star, rewards = createState(k)

    # N[i] represents number of times the action i was chosen
    N = np.zeros(k, dtype = np.uint32)

    # lists for plot
    optimalTaken=[]
    # Make moves
    for j in range(steps):
        a_greedy = np.argmax(Q)
        a_exploration = np.random.randint(0, Q.size)
        # Exploration probability
        a = np.random.choice([a_greedy, a_exploration], p = [1-e, e])

        optimal = np.argmax(rewards)
        if(a==optimal):
            optimalTaken.append(1)
        else:
            optimalTaken.append(0)

        N[a] = N[a] + 1

        stepSize = 1/N[a]
        Q[a] = updateEstimate(a, Q, rewards, stepSize)
        rewards = updateRewards(q_star, k, rewards)

    return(optimalTaken)

@ray.remote
def runGreedy(runs, steps):
    optimalGreedy = []
    # Greedy
    for i in range(runs):
        opSteps = bandit(k = 10, e = 0, steps = steps)
        optimalGreedy.append(opSteps)

        if((i+1)%100==0):
            print("Greedy: %d finished"%(i+1))

    optimalGreedy = np.array(optimalGreedy)
    optimalGreedy = np.sum(optimalGreedy, axis = 0)*100/runs

    return optimalGreedy

@ray.remote
def runExploration(runs, steps):
    optimalExploration = []
    # Explortory
    for i in range(runs):
        opSteps = bandit(k = 10, e = 0.1, steps = steps)
        optimalExploration.append(opSteps)
        if((i+1)%100==0):
            print("Exploration %d finished"%(i+1))

    optimalExploration = np.array(optimalExploration)
    optimalExploration = np.sum(optimalExploration, axis = 0)*100/runs

    return optimalExploration

def main():
    runs = 2000
    steps = 1000

    ret_id1 = runGreedy.remote(runs, steps)
    ret_id2 = runExploration.remote(runs, steps)

    optimalGreedy, optimalExploration = ray.get([ret_id1, ret_id2])

    plt.plot(optimalGreedy, color='red', label = "Greedy (e=0)")
    plt.plot(optimalExploration, color='blue', label = "Exploration (e=0.1)" )
    plt.xlabel('Steps')
    plt.ylabel('% of optimal actions')
    plt.ylim(ymin = 0)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
