import matplotlib.pyplot as plt

plt.ion()

def plot(scores, mean_scores):
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    plt.plot(scores)
    plt.plot(mean_scores)

    plt.legend(['Score', 'Mean Score'])

    plt.pause(0.1)