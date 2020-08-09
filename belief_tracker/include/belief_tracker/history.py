class History:
    """
    Container for collecting the history of observations, velocities and dts.
    """

    def __init__(self):
        self.observations = []
        self.velocities = []
        self.dts = []
        self.count = 0

    def size(self):
        return self.count

    def add_entry(self, observation, vel, dt):
        self.observations.append(observation)
        self.velocities.append(vel)
        self.dts.append(dt)
        self.count += 1

    def last_entry(self):
        if self.count > 0:
            last = self.count - 1
            return [self.observations[last], self.velocities[last], self.dts[last]]
        else:
            return None
