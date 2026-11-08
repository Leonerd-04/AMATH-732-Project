
class Particle:

    def __init__(self, x, y):

        self.x = x
        self.y = y

        # Infection state
        self.infected = False
        self.infection_timer = 0

        # Remember infection location
        self.x_sick = None
        self.y_sick = None

        # Mask state
        self.mask_on = False

    def infect(self, time_inf):

        self.infected = True
        self.infection_timer = time_inf

        # Record where infection occurred
        self.x_sick = self.x
        self.y_sick = self.y

    def update_infection_timer(self):

        if self.infection_timer > 0:
            self.infection_timer -= 1

        if self.infection_timer == 0:
            self.infected = False
