import numpy as np
import pygame
import os
import sys
import math
from math import e
from concurrent import futures
import ctypes
thread_pool_executor = futures.ThreadPoolExecutor(max_workers=1)

ctypes.windll.user32.SetProcessDPIAware()

class Program:
    pygame.init()
    clock = pygame.time.Clock()
    infoObject = pygame.display.Info()
    screen=pygame.display.set_mode((infoObject.current_w, infoObject.current_h),pygame.FULLSCREEN)
    number=0
    program_running=True
    started=False
    font = pygame.font.SysFont('consolas',bold=True, size=int(infoObject.current_w/120))
    _, _, filenames = next(os.walk("D:\PycharmProjects\\teste"))
    portofolii=list(filter(lambda name: name[len(name)-4:len(name)]==".txt" and name!="portofolii.txt",filenames))
    algoritmi=list(filter(lambda name: name[len(name)-3:len(name)]==".py" and name!="program.py",filenames))
    portofolii_poz=0
    algoritmi_poz=0
    algoritm_selectat=algoritmi[0]
    portofoliu_selectat=portofolii[0]
    pozitie_portofoliu_selectat=0
    pozitie_algoritm_selectat = 0
    playx=infoObject.current_w-infoObject.current_w/15
    playy=infoObject.current_h-infoObject.current_h/8.5
    rmed=[]
    randament_selectat=0.2
    randamente_poz = 0
    pozitie_randament_selectat=0
    bar_width=(playx-infoObject.current_w/30-10)/26
    bar_space=(playx-infoObject.current_w/30-10-20*bar_width)/21
    progress=0
    t=0
    maxGen=0

    def resource_path(relative_path):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    up_arrow = pygame.image.load(resource_path("up-arrow.png"))
    up_arrow.fill((100, 100, 150), special_flags=pygame.BLEND_RGB_SUB)
    down_arrow = pygame.image.load(resource_path("down-arrow.png"))
    down_arrow.fill((100, 100, 150), special_flags=pygame.BLEND_RGB_SUB)
    up_arrow_hover = pygame.image.load(resource_path("up-arrow.png"))
    up_arrow_hover.fill((0, 0, 100), special_flags=pygame.BLEND_RGB_ADD)
    down_arrow_hover = pygame.image.load(resource_path("down-arrow.png"))
    down_arrow_hover.fill((0, 0, 100), special_flags=pygame.BLEND_RGB_ADD)
    left_arrow = pygame.image.load(resource_path("left-arrow.png"))
    left_arrow.fill((100, 100, 50), special_flags=pygame.BLEND_RGB_SUB)
    left_arrow_hover = pygame.image.load(resource_path("left-arrow.png"))
    left_arrow_hover.fill((100, 100, 50), special_flags=pygame.BLEND_RGB_SUB)
    left_arrow_hover.fill((0, 100, 255), special_flags=pygame.BLEND_RGB_ADD)
    right_arrow = pygame.image.load(resource_path("right-arrow.png"))
    right_arrow.fill((100, 100, 50), special_flags=pygame.BLEND_RGB_SUB)
    right_arrow_hover = pygame.image.load(resource_path("right-arrow.png"))
    right_arrow_hover.fill((100, 100, 50), special_flags=pygame.BLEND_RGB_SUB)
    right_arrow_hover.fill((0, 100, 255), special_flags=pygame.BLEND_RGB_ADD)
    play = pygame.image.load(resource_path("play.png"))
    play.fill((11, 118, 212), special_flags=pygame.BLEND_RGB_ADD)
    play_hover=pygame.image.load(resource_path("play.png"))
    play_hover.fill((0, 145, 31), special_flags=pygame.BLEND_RGB_ADD)
    stop = pygame.image.load(resource_path("stop.png"))
    stop.fill((11, 118, 212), special_flags=pygame.BLEND_RGB_ADD)
    stop_hover = pygame.image.load(resource_path("stop.png"))
    stop_hover.fill((196, 2, 2), special_flags=pygame.BLEND_RGB_ADD)
    cog = pygame.image.load(resource_path("cog.png"))
    cog.fill((174, 189, 214), special_flags=pygame.BLEND_RGB_ADD)
    angle=0
    running=False
    thread_running=True
    vector=[]
    max_port_size=0
    for i in range(portofolii_poz, portofolii_poz + 3, 1):
        text_width, text_height = font.size(portofolii[i])
        if text_width>max_port_size:
            max_port_size=text_width
    max_alg_size=0
    for i in range(algoritmi_poz, algoritmi_poz + 3, 1):
        text_width, text_height = font.size(algoritmi[i])
        if text_width>max_alg_size:
            max_alg_size=text_width
    max_rand_size = 0
    best_portfolio=[]

    def display_title(self):
        self.screen.fill((11, 118, 212), (0, 0, self.infoObject.current_w, self.infoObject.current_h / 20))
        text_width, text_height = self.font.size("TEHNICI DE CALCUL EVOLUTIV PENTRU OPTIMIZAREA PORTOFOLIILOR DE ACTIUNI.")
        text_display = self.font.render("TEHNICI DE CALCUL EVOLUTIV PENTRU OPTIMIZAREA PORTOFOLIILOR DE ACTIUNI.", True,
                                        (255, 255, 255))
        self.screen.blit(text_display, (self.infoObject.current_w / 2-text_width/2, self.infoObject.current_h / 40-text_height/2))
        surf = pygame.transform.rotate(self.cog, self.angle)
        self.screen.blit(surf, (self.infoObject.current_w/30, self.infoObject.current_h/120))
        if self.angle>=360:
            self.angle=0
        self.angle+=0.1

    def citeste_date(self,nume):
        R = np.genfromtxt(nume)
        # R=generate_matrix()
        # print(R)
        nr_obs = R.shape[0]
        nr_actiuni = R.shape[1]
        B = -np.ones([nr_actiuni, nr_actiuni - 1])
        B[:nr_actiuni - 1, :nr_actiuni - 1] = np.eye(nr_actiuni - 1)
        alpha = np.zeros([nr_actiuni, 1])
        alpha[nr_actiuni - 1] = 1
        rmed = np.mean(R, axis=0)
        rmed = np.reshape(rmed, [nr_actiuni, 1])
        # matricea de covarianta - estimatie nedeplasata
        Q = np.cov(R.T) * (nr_obs - 1) / nr_obs
        return Q, rmed, alpha, B, nr_actiuni

    def fobiectiv(self,Q, rmed, B, alpha, ro, Rp, x):
        y = alpha + np.matmul(B, x)
        # np.matmul(a,b)=a*b - inmultirea a doua matrice
        y1 = np.matmul(y.T, Q)
        V = np.matmul(y1, y)
        c = ro / (Rp * Rp)
        y = np.matmul(rmed.T, alpha)
        y1 = np.matmul(rmed.T, B)
        y2 = np.matmul(y1, x)
        val = V + c * np.square(y - Rp + y2)
        return val, V

    def gen_flower(self,nr_actiuni):
        flower = np.zeros(nr_actiuni - 1)
        s = 0
        i = 0
        generated = np.zeros(nr_actiuni - 1)
        while i < nr_actiuni - 1 and s < 1:
            value = np.random.uniform(0, 1)
            j = np.random.randint(0, nr_actiuni - 1)
            while generated[j]:
                j = np.random.randint(0, nr_actiuni - 1)
            if value + s <= 1:
                generated[j] = 1
                flower[j] = value
                s = s + value
            else:
                flower[j] = 1 - s
                s = 1
            i += 1
        return flower

    def gen_pop(self,dim, nr_actiuni, Q, rmed, B, alpha, ro, Rp):
        pop = np.zeros([dim, nr_actiuni])
        for i in range(dim):
            x = self.gen_flower(nr_actiuni)
            pop[i][:nr_actiuni - 1] = x
            ps = x[:nr_actiuni - 1]
            y = ps.reshape([nr_actiuni - 1, 1])
            val, V = self.fobiectiv(Q, rmed, B, alpha, ro, Rp, y)
            pop[i][nr_actiuni - 1] = 1 / (1 + val)
        return pop

    def draw_Levy_distribution_vector(self,nr_actiuni, Lambda):
        L = np.zeros(nr_actiuni - 1)
        for i in range(nr_actiuni - 1):
            gamma = np.random.gamma(Lambda)
            L[i] = (Lambda * gamma * np.sin((math.pi * Lambda) / 2) / math.pi * pow(0.5, 1 + Lambda))
        return L

    def draw_from_uniform_distribution(self,nr_actiuni):
        E = np.zeros(nr_actiuni - 1)
        for i in range(nr_actiuni - 1):
            E[i] = np.random.uniform(0, 1)
        return E

    def global_polination(self,nr_actiuni, Q, rmed, B, alpha, ro, Rp, L, flower, g):
        new_flower = np.zeros(nr_actiuni)
        for i in range(nr_actiuni - 1):
            new_flower[i] = flower[i] + L[i] * (g[i] - flower[i])
            if new_flower[i] < 0:
                new_flower[i] = 0
        if np.sum(new_flower[:nr_actiuni - 1]) > 1:
            sum = np.sum(new_flower[:nr_actiuni - 1]) - 1
            while sum > 0:
                poz = np.random.randint(nr_actiuni - 1)
                if new_flower[:nr_actiuni - 1][poz] > sum:
                    new_flower[:nr_actiuni - 1][poz] = new_flower[:nr_actiuni - 1][poz] - sum
                    sum = 0
                else:
                    sum = sum - new_flower[:nr_actiuni - 1][poz]
                    new_flower[:nr_actiuni - 1][poz] = 0
        ps = new_flower[:nr_actiuni - 1]
        y = ps.reshape([nr_actiuni - 1, 1])
        val, V = self.fobiectiv(Q, rmed, B, alpha, ro, Rp, y)
        new_flower[nr_actiuni - 1] = 1 / (1 + val)
        return new_flower

    def local_polination(self,nr_actiuni, Q, rmed, B, alpha, ro, Rp, E, flower, flower1, flower2):
        new_flower = np.zeros(nr_actiuni)
        for i in range(nr_actiuni - 1):
            new_flower[i] = flower[i] + E[i] * (flower1[i] - flower2[i])
            if new_flower[i] < 0:
                new_flower[i] = 0
        if np.sum(new_flower[:nr_actiuni - 1]) > 1:
            sum = np.sum(new_flower[:nr_actiuni - 1]) - 1
            while sum > 0:
                poz = np.random.randint(nr_actiuni - 1)
                if new_flower[:nr_actiuni - 1][poz] > sum:
                    new_flower[:nr_actiuni - 1][poz] = new_flower[:nr_actiuni - 1][poz] - sum
                    sum = 0
                else:
                    sum = sum - new_flower[:nr_actiuni - 1][poz]
                    new_flower[:nr_actiuni - 1][poz] = 0
        ps = new_flower[:nr_actiuni - 1]
        y = ps.reshape([nr_actiuni - 1, 1])
        val, V = self.fobiectiv(Q, rmed, B, alpha, ro, Rp, y)
        new_flower[nr_actiuni - 1] = 1 / (1 + val)
        return new_flower

    def move_firefly(self,better, worse, nr_actiuni, Q, rmed, B, alpha, ro, Rp, gamma, attractiveness, randomization_factor):
        RHS = np.zeros(nr_actiuni - 1)
        random = np.zeros(nr_actiuni - 1)
        for i in range(nr_actiuni - 1):
            random[i] = randomization_factor * np.random.uniform(-1, 1)
        for i in range(nr_actiuni - 1):
            RHS[i] = attractiveness * pow(e, -gamma * pow(worse[i] - better[i], 2)) * (better[i] - worse[i])
        intermediar = np.add(worse[:nr_actiuni - 1], RHS)
        rez = np.add(intermediar, random)
        for i in range(nr_actiuni - 1):
            if rez[i] < 0:
                rez[i] = 0
        if np.sum(rez) > 1:
            sum = np.sum(rez) - 1
            while sum > 0:
                poz = np.random.randint(nr_actiuni - 1)
                if rez[poz] > sum:
                    rez[poz] = rez[poz] - sum
                    sum = 0
                else:
                    sum = sum - rez[poz]
                    rez[poz] = 0
        worse[:nr_actiuni - 1] = rez
        y = rez.reshape([nr_actiuni - 1, 1])
        val, V = self.fobiectiv(Q, rmed, B, alpha, ro, Rp, y)
        worse[nr_actiuni - 1] = 1 / (1 + val)
        return worse

    def sortare_populatie(self,pop, nr_actiuni, dim):
        for i in range(dim - 1):
            for j in range(i, dim, 1):
                if pop[i][nr_actiuni - 1] < pop[j][nr_actiuni - 1]:
                    x = pop[i][nr_actiuni - 1]
                    pop[i][nr_actiuni - 1] = pop[j][nr_actiuni - 1]
                    pop[j][nr_actiuni - 1] = x
        return pop


    def FPA_initialization(self):
        self.vector = []
        self.Q, self.rmed1, self.alpha, self.B, self.nr_actiuni = self.citeste_date(self.portofoliu_selectat)
        self.ro = 10
        self.Rp = self.randament_selectat
        self.dim = 50
        self.switch_propability = 0.4
        self.Lambda = 1.5
        self.t = 0
        self.maxGen = 500
        self.next_bar = self.maxGen / 20
        self.pop = self.gen_pop(self.dim, self.nr_actiuni, self.Q, self.rmed1, self.B, self.alpha, self.ro, self.Rp)
        self.g = self.pop[0]
        self.history = np.zeros([self.maxGen + 1, self.nr_actiuni])
        self.history[self.t] = self.g
        for i in range(self.dim):
            if self.pop[i][self.nr_actiuni - 1] > self.g[self.nr_actiuni - 1]:
                self.g = self.pop[i]


    def HSFA_initialization(self):
        self.vector = []
        self.dim = 50
        self.ro = 10
        self.Rp = self.randament_selectat
        self.maxGen = 500
        self.next_bar = self.maxGen / 20
        self.t = 0
        self.Q, self.rmed1, self.alpha, self.B, self.nr_actiuni = self.citeste_date(self.portofoliu_selectat)
        self.pop = self.gen_pop(self.dim, self.nr_actiuni, self.Q, self.rmed1, self.B, self.alpha, self.ro, self.Rp)
        self.pop = self.sortare_populatie(self.pop, self.nr_actiuni, self.dim)
        self.history = np.zeros([self.maxGen + 1, self.nr_actiuni])
        self.history[self.t] = self.pop[0]
        self.keep = self.dim // 10
        self.gamma = 1
        self.randomization_factor = 2
        self.HMCR = 0.9
        self.PAR = 0.1
        self.attractiveness = 1

    def FA_initialization(self):
        self.vector = []
        self.dim = 24
        self.ro = 10
        self.Rp = self.randament_selectat
        self.maxGen = 300
        self.next_bar = self.maxGen / 20
        self.t = 0
        self.Q, self.rmed1, self.alpha, self.B, self.nr_actiuni = self.citeste_date(self.portofoliu_selectat)
        self.pop = self.gen_pop(self.dim, self.nr_actiuni, self.Q, self.rmed1, self.B, self.alpha, self.ro, self.Rp)
        self.pop = self.sortare_populatie(self.pop, self.nr_actiuni, self.dim)
        self.history = np.zeros([self.maxGen + 1, self.nr_actiuni])
        self.history[self.t] = self.pop[0]
        self.gamma = 1
        self.randomization_factor = 2
        self.attractiveness = 1

    def HSFA(self):
        if self.t < self.maxGen:
            for i in range(self.keep):
                for j in range(self.dim):
                    if self.pop[j][self.nr_actiuni - 1] < self.pop[i][self.nr_actiuni - 1]:
                        self.pop[j] = self.move_firefly(self.pop[i], self.pop[j], self.nr_actiuni, self.Q, self.rmed1, self.B, self.alpha, self.ro, self.Rp, self.gamma,
                                              self.attractiveness, self.randomization_factor)
                    else:
                        for k in range(self.nr_actiuni - 1):
                            rand = np.random.uniform(0, 1)
                            if rand < self.HMCR:
                                r1 = int(self.dim * rand)
                                self.pop[i][k] = self.pop[r1][k]
                                rand1 = np.random.uniform(-1, 1)
                                if rand1 < self.PAR:
                                    self.pop[i][k] = self.pop[i][k] + rand1 / 2
                            else:
                                self.pop[i][k] = self.pop[self.dim - 1][k] + rand * (self.pop[0][k] - self.pop[self.dim - 1][k])
                            if self.pop[i][k] < 0:
                                self.pop[i][k] = 0
                        if np.sum(self.pop[i][:self.nr_actiuni - 1]) > 1:
                            sum = np.sum(self.pop[i][:self.nr_actiuni - 1]) - 1
                            while sum > 0:
                                poz = np.random.randint(self.nr_actiuni - 1)
                                if self.pop[i][:self.nr_actiuni - 1][poz] > sum:
                                    self.pop[i][:self.nr_actiuni - 1][poz] = self.pop[i][:self.nr_actiuni - 1][poz] - sum
                                    sum = 0
                                else:
                                    sum = sum - self.pop[i][:self.nr_actiuni - 1][poz]
                                    self.pop[i][:self.nr_actiuni - 1][poz] = 0
                        ps = self.pop[i][:self.nr_actiuni - 1]
                        y = ps.reshape([self.nr_actiuni - 1, 1])
                        val, V = self.fobiectiv(self.Q, self.rmed1, self.B, self.alpha, self.ro, self.Rp, y)
                        self.pop[i][self.nr_actiuni - 1] = 1 / (1 + val)
            self.pop = self.sortare_populatie(self.pop, self.nr_actiuni, self.dim)
            self.randomization_factor = self.randomization_factor * 0.96
            if self.history[self.t][self.nr_actiuni - 1] >= self.pop[0][self.nr_actiuni - 1]:
                self.history[self.t + 1] = self.history[self.t]
            else:
                self.history[self.t + 1] = self.pop[0]
            self.t += 1
            self.vector.append(self.history[self.t][self.nr_actiuni - 1])
            self.best_portfolio=[]
            for i in range(self.nr_actiuni):
                self.best_portfolio.append(self.history[self.t][i])
            if self.t == self.next_bar:
                self.progress += 1
                self.next_bar += self.maxGen / 20
        self.history_v = np.zeros(self.t)
        for i in range(self.t):
            self.history_v[i] = self.history[i][self.nr_actiuni - 1]
        if self.t >= self.maxGen:
            self.running = False


    def FA(self):
        if self.t < self.maxGen:
            for i in range(self.dim):
                for j in range(self.dim):
                    if self.pop[j][self.nr_actiuni - 1] < self.pop[i][self.nr_actiuni - 1]:
                        self.pop[j] = self.move_firefly(self.pop[i], self.pop[j], self.nr_actiuni, self.Q, self.rmed1, self.B, self.alpha, self.ro, self.Rp, self.gamma,
                                              self.attractiveness, self.randomization_factor)
            self.randomization_factor = self.randomization_factor * 0.96
            if self.history[self.t][self.nr_actiuni - 1] >= self.pop[0][self.nr_actiuni - 1]:
                self.history[self.t + 1] = self.history[self.t]
            else:
                self.history[self.t + 1] = self.pop[0]
            self.t += 1
            self.vector.append(self.history[self.t][self.nr_actiuni - 1])
            self.best_portfolio = []
            for i in range(self.nr_actiuni):
                self.best_portfolio.append(self.history[self.t][i])
            if self.t == self.next_bar:
                self.progress += 1
                self.next_bar += self.maxGen / 20
        history_v = np.zeros(self.t)
        for i in range(self.t):
            history_v[i] = self.history[i][self.nr_actiuni - 1]
        if self.t >= self.maxGen:
            self.running = False


    def FPA(self):
            if self.t < self.maxGen and self.running is True:
                for i in range(self.dim):
                    random = np.random.uniform(0, 1)
                    if random <= self.switch_propability:
                        L = self.draw_Levy_distribution_vector(self.nr_actiuni, self.Lambda)
                        new_flower = self.global_polination(self.nr_actiuni, self.Q, self.rmed1, self.B, self.alpha, self.ro, self.Rp, L, self.pop[i], self.g)
                    else:
                        E = self.draw_from_uniform_distribution(self.nr_actiuni)
                        poz = np.random.randint(0, self.dim)
                        poz1 = np.random.randint(0, self.dim)
                        while poz1 == poz:
                            poz1 = np.random.randint(0, self.dim)
                        new_flower = self.local_polination(self.nr_actiuni, self.Q, self.rmed1, self.B, self.alpha, self.ro, self.Rp, E, self.pop[i], self.pop[poz], self.pop[poz1])
                    if new_flower[self.nr_actiuni - 1] > self.pop[i][self.nr_actiuni - 1]:
                        self.pop[i] = new_flower
                self.g = self.pop[0]
                for i in range(self.dim):
                    if self.pop[i][self.nr_actiuni - 1] > self.g[self.nr_actiuni - 1]:
                        self.g = self.pop[i]
                if self.history[self.t][self.nr_actiuni - 1] >= self.pop[0][self.nr_actiuni - 1]:
                    self.history[self.t + 1] = self.history[self.t]
                else:
                    self.history[self.t + 1] = self.g
                self.t += 1
                self.vector.append(self.history[self.t][self.nr_actiuni - 1])
                self.best_portfolio = []
                for i in range(self.nr_actiuni):
                    self.best_portfolio.append(self.history[self.t][i])
                if self.t==self.next_bar:
                    self.progress+=1
                    self.next_bar+=self.maxGen/20
            #history_v = np.zeros(self.t)
            # for i in range(self.t):
            #     history_v[i] = self.history[i][self.nr_actiuni - 1]
            if self.t>=self.maxGen:
                self.running = False

    def display_arrow(self,arrow,x,y):
        self.screen.blit(arrow, (x,y))

    def display_text_actiuni(self,text,x,y):
        if float(text)<0.01:
            text_display = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(text_display, (x, y))
        if float(text)>=0.01 and float(text)<=0.1:
            text_display = self.font.render(text, True, (150, 0, 0))
            self.screen.blit(text_display, (x, y))
        if float(text)>0.1 and float(text)<=0.2:
            text_display = self.font.render(text, True, (150, 100, 0))
            self.screen.blit(text_display, (x, y))
        if float(text)>0.2 and float(text)<=0.4:
            text_display = self.font.render(text, True, (120, 120, 0))
            self.screen.blit(text_display, (x, y))
        if float(text)>0.4:
            text_display = self.font.render(text, True, (0, 120, 0))
            self.screen.blit(text_display, (x, y))

    def display_text(self,text,x,y):
        text_display = self.font.render(text, True,(255, 255, 255))
        self.screen.blit(text_display,(x,y))

    def display_text_blue(self,text,x,y):
        text_display = self.font.render(text, True,(0, 200, 200))
        self.screen.blit(text_display,(x,y))

    def display_text_green(self,text,x,y):
        text_display = self.font.render(text, True,(0, 0, 255))
        self.screen.blit(text_display,(x,y))

    def draw_arrows(self,x,y,y1,mx,my):
        if mx > x and mx < x+32 and my > y and my < y + 32:
            self.display_arrow(self.up_arrow_hover, x, y)
        else:
            self.display_arrow(self.up_arrow, x, y)
        if mx > x and mx < x+32 and my > y1 and my < y1 + 32:
            self.display_arrow(self.down_arrow_hover, x, y1)
        else:
            self.display_arrow(self.down_arrow, x, y1)

    def display_play_stop(self,mx,my):
            if mx > self.playx and mx < self.playx + 64 and my > self.playy and my < self.playy + 64:
                if self.running==False:
                    self.screen.blit(self.play_hover, (self.playx, self.playy))
                else:
                    self.screen.blit(self.stop_hover, (self.playx, self.playy))
            else:
                if self.running == False:
                    self.screen.blit(self.play, (self.playx, self.playy))
                else:
                    self.screen.blit(self.stop, (self.playx, self.playy))

    def display_progress(self):
        if self.running is True:
            pygame.draw.rect(self.screen, (149, 162, 184), pygame.Rect(self.infoObject.current_w/30-2*self.infoObject.current_w/192+self.bar_space, self.playy - 15,20 * self.bar_space + 20 * self.bar_width,94))
            pygame.draw.rect(self.screen, (11, 118, 212), pygame.Rect(self.infoObject.current_w/30-self.infoObject.current_w/192+self.bar_space - self.infoObject.current_w/192, self.playy - 15,20 * self.bar_space + 20 * self.bar_width, 94), 4)
            for i in range(self.progress):
                if self.progress<20:
                    pygame.draw.rect(self.screen, (11, 118, 212), pygame.Rect(self.infoObject.current_w/30-self.infoObject.current_w/192+self.bar_space*(i+1)+self.bar_width*i, self.playy, self.bar_width, 64))
        else:
            if self.progress == 20:
                pygame.draw.rect(self.screen, (149, 162, 184), pygame.Rect(self.infoObject.current_w/30-2*self.infoObject.current_w/192+self.bar_space, self.playy - 15,
                                                                           20 * self.bar_space + 20 * self.bar_width,
                                                                           94))
                pygame.draw.rect(self.screen, (11, 118, 212), pygame.Rect(self.infoObject.current_w/30-2*self.infoObject.current_w/192+self.bar_space, self.playy - 15,
                                                                          20 * self.bar_space + 20 * self.bar_width,
                                                                          94), 4)
                for i in range(self.progress):
                    pygame.draw.rect(self.screen, (0, 145, 31),pygame.Rect(self.infoObject.current_w/30-self.infoObject.current_w/192+self.bar_space * (i + 1) + self.bar_width * i, self.playy,self.bar_width, 64))

    def genereaza_randamente(self):
        self.rmed=[]
        Q, rmed, alpha, B, nr_actiuni = self.citeste_date(self.portofoliu_selectat)
        rmed = np.reshape(rmed, [1, nr_actiuni])
        max=np.max(rmed)
        min=np.min(rmed)
        dif=max-min
        med=0
        for i in range(len(rmed[0])):
            med+=rmed[0][i]
        med=med/len(rmed[0])
        k=-0.4*(dif/med)
        no=0
        gata=False
        while no<10 and gata==False:
            if np.round(med+k*med,2)>max:
                gata=True
            else:
                if np.round(med+k*med,2)>0:
                    self.rmed.append(np.round(med+k*med,2))
                    no+=1
            k+=0.1*(dif/med)
        self.randament_selectat=self.rmed[0]
        self.pozitie_randament_selectat=0

    def run_algorithm(self):
        if self.running is True:
            if self.algoritm_selectat == "FPA.py":
                self.FPA()
            if self.algoritm_selectat == "HSFA.py":
                self.HSFA()
            if self.algoritm_selectat == "FA.py":
                self.FA()

    def initialization(self):
        if self.algoritm_selectat == "FPA.py":
            self.FPA_initialization()
        if self.algoritm_selectat == "HSFA.py":
            self.HSFA_initialization()
        if self.algoritm_selectat == "FA.py":
            self.FA_initialization()

    def display_plot(self):
        x=self.infoObject.current_w / 5 + self.max_alg_size + self.max_port_size+64
        y=self.infoObject.current_h / 20 + self.infoObject.current_h / 17-8
        width=self.infoObject.current_w-x-self.infoObject.current_w/15+64
        height=self.infoObject.current_h-y-self.infoObject.current_h/8.5-32
        pygame.draw.rect(self.screen, (149, 162, 184),pygame.Rect(x, y, width, height))
        pygame.draw.rect(self.screen, (11, 118, 212), pygame.Rect(x-2, y-2, width+4, height+4),4)
        if self.running is True:
            maxim=max(self.vector)
            self.display_text_green("fob value",x + width / 30,y + height / 20-30)
            self.display_text(str(np.round(maxim,2)), x + width / 30, y + height / 20)
            self.display_text(str(np.round(self.vector[0] + (maxim - self.vector[0]) / 2 + (maxim - self.vector[0]) / 4, 2)),
                              x + width / 30, (y + height / 20 + (y + height - height / 20 + y + height / 20) / 2) / 2)
            self.display_text(str(np.round(self.vector[0] + (maxim - self.vector[0]) / 2, 2)), x + width / 30,
                              (y + height - height / 20 + y + height / 20) / 2)
            self.display_text(str(np.round(self.vector[0] + (maxim - self.vector[0]) / 4, 2)), x + width / 30,
                              (y + height - height / 20 + (y + height - height / 20 + y + height / 20) / 2) / 2)
            self.display_text(str(np.round(self.vector[0], 2)), x + width / 30, y + height - height / 20)
        text_width, text_height = self.font.size("0.00")
        pygame.draw.line(self.screen, (180, 191, 209), (x + width / 30 + text_width + 5, y + height / 6),(x + width / 30 + text_width + 5 + width-2*width/30, y + height / 6))
        pygame.draw.line(self.screen, (180, 191, 209), (x + width / 30 + text_width + 5, y + height * 2/6),(x + width / 30 + text_width + 5 + width - 2 * width / 30, y + height * 2/6))
        pygame.draw.line(self.screen, (180, 191, 209), (x + width / 30 + text_width + 5, y + height * 3/6),(x + width / 30 + text_width + 5 + width - 2 * width / 30, y + height * 3/6))
        pygame.draw.line(self.screen, (180, 191, 209), (x + width / 30 + text_width + 5, y + height * 4/6),(x + width / 30 + text_width + 5 + width - 2 * width / 30, y + height * 4/6))
        pygame.draw.line(self.screen, (180, 191, 209), (x + width / 30 + text_width + 5, y + height * 5/6),(x + width / 30 + text_width + 5 + width - 2 * width / 30, y + height * 5/6))
        pygame.draw.line(self.screen, (106, 116, 128), (x + width / 30+text_width+5, y +  height / 20),(x+width/30+text_width+5,y+height-height/20),5)
        pygame.draw.line(self.screen, (106, 116, 128), (x + width / 30 + text_width + 5, y+height-height/20),(x + width-width/30, y+height-height/20), 5)
        if self.running is True:
            miny=y +  height / 20
            maxy=y+height-height/20-15
            minx=x + width / 30 + text_width + 15
            maxx=x + width-width/30
            self.display_text("0", minx, maxy + 30)
            self.display_text(str(self.maxGen // 5), (1 / 5 * (maxx - minx) + minx), maxy + 30)
            self.display_text(str((self.maxGen // 5) * 2), (2 / 5 * (maxx - minx) + minx), maxy + 30)
            self.display_text(str((self.maxGen // 5) * 3), (3 / 5 * (maxx - minx) + minx), maxy + 30)
            self.display_text(str((self.maxGen // 5) * 4), (4 / 5 * (maxx - minx) + minx), maxy + 30)
            self.display_text(str((self.maxGen // 5) * 5), (5 / 5 * (maxx - minx) + minx), maxy + 30)
            text_width, text_height = self.font.size("generatie")
            self.display_text_green("generatie", (5 / 5 * (maxx - minx) + minx)-text_width, maxy - 30)
            for i in range(0,self.t,1):
                if i > 0:
                    pygame.draw.line(self.screen, (66, 135, 245), ((i - 1) / self.maxGen * (maxx - minx) + minx,miny + ((maxim-self.vector[i-1]) / (maxim-self.vector[0]))  * (maxy - miny)), (i / self.maxGen * (maxx - minx) + minx,miny + ((maxim-self.vector[i]) / (maxim-self.vector[0]))  * (maxy - miny)), 5)
                pygame.draw.circle(self.screen, (39, 118, 245), (i / self.maxGen * (maxx - minx) + minx,miny + ((maxim-self.vector[i]) / (maxim-self.vector[0]))  * (maxy - miny)), 5)
        if self.t>0 and self.t==self.maxGen:
            x = self.infoObject.current_w / 5 + self.max_alg_size + self.max_port_size + 64
            y = self.infoObject.current_h / 20 + self.infoObject.current_h / 17
            width = self.infoObject.current_w - x - self.infoObject.current_w / 15 + 64
            height = self.infoObject.current_h - y - self.infoObject.current_h / 8.5 - 32
            self.display_text_green("fob value", x + width / 30, y + height / 20 - 30)
            if True:
                maxim = max(self.vector)
                self.display_text(str(np.round(maxim, 2)), x + width / 30, y + height / 20)
                self.display_text(
                    str(np.round(self.vector[0] + (maxim - self.vector[0]) / 2 + (maxim - self.vector[0]) / 4, 2)),
                    x + width / 30, (y + height / 20 + (y + height - height / 20 + y + height / 20) / 2) / 2)
                self.display_text(str(np.round(self.vector[0] + (maxim - self.vector[0]) / 2, 2)), x + width / 30,
                                  (y + height - height / 20 + y + height / 20) / 2)
                self.display_text(str(np.round(self.vector[0] + (maxim - self.vector[0]) / 4, 2)), x + width / 30,
                                  (y + height - height / 20 + (y + height - height / 20 + y + height / 20) / 2) / 2)
                self.display_text(str(np.round(self.vector[0], 2)), x + width / 30, y + height - height / 20)
            text_width, text_height = self.font.size("0.00")
            pygame.draw.line(self.screen, (106, 116, 128), (x + width / 30 + text_width + 5, y + height / 20),
                             (x + width / 30 + text_width + 5, y + height - height / 20), 5)
            pygame.draw.line(self.screen, (106, 116, 128), (x + width / 30 + text_width + 5, y + height - height / 20),
                             (x + width - width / 30, y + height - height / 20), 5)
            miny = y + height / 20
            maxy = y + height - height / 20 - 15
            minx = x + width / 30 + text_width + 15
            maxx = x + width - width / 30
            self.display_text("0", minx, maxy + 30)
            self.display_text(str(self.maxGen // 5), (1 / 5 * (maxx - minx) + minx), maxy + 30)
            self.display_text(str((self.maxGen // 5)*2), (2 / 5 * (maxx - minx) + minx), maxy + 30)
            self.display_text(str((self.maxGen // 5)*3), (3 / 5 * (maxx - minx) + minx), maxy + 30)
            self.display_text(str((self.maxGen // 5)*4), (4 / 5 * (maxx - minx) + minx), maxy + 30)
            self.display_text(str((self.maxGen // 5)*5), (5 / 5 * (maxx - minx) + minx), maxy + 30)
            text_width, text_height = self.font.size("generatie")
            self.display_text_green("generatie", (5 / 5 * (maxx - minx) + minx) - text_width, maxy - 30)
            for i in range(0, self.t, 1):
                if i > 0:
                    pygame.draw.line(self.screen, (66, 135, 245), ((i - 1) / self.maxGen * (maxx - minx) + minx,
                                                                   miny + ((maxim - self.vector[i - 1]) / (
                                                                               maxim - self.vector[0])) * (
                                                                               maxy - miny)), (
                                     i / self.maxGen * (maxx - minx) + minx,
                                     miny + ((maxim - self.vector[i]) / (maxim - self.vector[0])) * (maxy - miny)), 5)
                pygame.draw.circle(self.screen, (39, 118, 245), (i / self.maxGen * (maxx - minx) + minx, miny + (
                            (maxim - self.vector[i]) / (maxim - self.vector[0])) * (maxy - miny)), 5)

    def display_bgs(self):
        pygame.draw.rect(self.screen, (11, 118, 212), pygame.Rect(self.infoObject.current_w/30-10, self.infoObject.current_h / 20+self.infoObject.current_h/17-10, self.infoObject.current_w/5 + self.max_alg_size+self.max_port_size-self.infoObject.current_w/30+10+self.max_rand_size+10, 4* self.infoObject.current_h / 17+52), 4)
        pygame.draw.rect(self.screen, (149, 162, 184), pygame.Rect(self.infoObject.current_w / 30 - 10+2,
                                                                  self.infoObject.current_h / 20 + self.infoObject.current_h / 17 - 10+2,
                                                                  self.infoObject.current_w / 5 + self.max_alg_size + self.max_port_size - self.infoObject.current_w / 30 + self.max_rand_size + 16,
                                                                  4 * self.infoObject.current_h / 17 + 52-4))

    def get_actiuni_portofolii(self):
        self.actiuni=[]
        self.actiuni_poz=0
        self.started=True
        f=open("portofolii.txt", "r")
        for line in f:
            cuvinte=line.split(",")
            if cuvinte[0]==self.portofoliu_selectat:
                self.nr_actiuni=int(cuvinte[1])
                for i in range(2,self.nr_actiuni+2):
                    self.actiuni.append(cuvinte[i])

    def display_values(self,mx,my):
        pygame.draw.rect(self.screen, (149, 162, 184), pygame.Rect(self.infoObject.current_w/30-10,
                                                                   self.infoObject.current_h / 20 + self.infoObject.current_h / 17 - 8+4 * self.infoObject.current_h / 17 +63,
                                                                   self.infoObject.current_w / 5 + self.max_alg_size + self.max_port_size - self.infoObject.current_w / 30 + self.max_rand_size + 16,
                                                                   self.playy-(self.infoObject.current_h / 20 + self.infoObject.current_h / 17 - 8+4 * self.infoObject.current_h / 17 +63)-30))
        pygame.draw.rect(self.screen, (11, 118, 212), pygame.Rect(self.infoObject.current_w / 30 - 10,
                                                                   self.infoObject.current_h / 20 + self.infoObject.current_h / 17 - 8 + 4 * self.infoObject.current_h / 17 + 63,
                                                                   self.infoObject.current_w / 5 + self.max_alg_size + self.max_port_size - self.infoObject.current_w / 30 + self.max_rand_size + 16+2,
                                                                   self.playy - (
                                                                               self.infoObject.current_h / 20 + self.infoObject.current_h / 17 - 8 + 4 * self.infoObject.current_h / 17 + 63) - 30),int(self.infoObject.current_w/480))

        self.display_text("PORTOFOLIU",(self.infoObject.current_w/30-10+self.infoObject.current_w / 5 + self.max_alg_size + self.max_port_size - self.infoObject.current_w / 30 + self.max_rand_size + 16-self.infoObject.current_w/30-10)/2,self.infoObject.current_h / 20 + self.infoObject.current_h / 17 - 8+4 * self.infoObject.current_h / 17 +63+15)

        arrow1_pozx=self.infoObject.current_w/30
        arrow1_pozy=self.infoObject.current_h / 20 + self.infoObject.current_h / 17 - 8+6 * self.infoObject.current_h / 17 +63+15
        arrow2_pozx=self.infoObject.current_w / 30-10+self.infoObject.current_w / 5 + self.max_alg_size + self.max_port_size - self.infoObject.current_w / 30 + self.max_rand_size+16-32-10
        arrow2_pozy=self.infoObject.current_h / 20 + self.infoObject.current_h / 17 - 8 + 6 * self.infoObject.current_h / 17 + 63 + 15
        text_width, text_height = self.font.size("MMMM")
        space = arrow2_pozx - arrow1_pozx - 4 * text_width
        text_space = space / 5
        k = 0

        if self.started is True:
            x = np.array(self.best_portfolio[:self.nr_actiuni-1])
            x1=x.reshape([self.nr_actiuni-1, 1])
            val, V = self.fobiectiv(self.Q, self.rmed1, self.B, self.alpha, self.ro, self.Rp, x1)
            Rr = np.matmul((self.rmed1.T), (self.alpha + np.matmul(self.B, x1)))[0][0]
            for i in range(self.actiuni_poz, self.actiuni_poz + 4, 1):
                if self.nr_actiuni>i:
                    self.display_text(self.actiuni[i], arrow1_pozx + (k + 1) * text_space + k * text_width, arrow1_pozy - self.infoObject.current_w/96)
                    k += 1
            for i in range(self.nr_actiuni - 1):
                self.best_portfolio[i] = round(self.best_portfolio[i], 2)
            self.best_portfolio[self.nr_actiuni - 1] = round(1 - sum(self.best_portfolio[:self.nr_actiuni - 1]), 2)
            if self.best_portfolio[self.nr_actiuni - 1]<0:
                self.best_portfolio[self.nr_actiuni - 1]=0.0
            k=0
            for i in range(self.actiuni_poz, self.actiuni_poz + 4, 1):
                if self.nr_actiuni>i:
                    self.display_text_actiuni(str(self.best_portfolio[i]), arrow1_pozx + (k + 1) * text_space + k * text_width, arrow1_pozy +32+ self.infoObject.current_w/96)
                    k += 1

            text_width, text_height = self.font.size("RANDAMENTUL OBTINUT")
            subtitlu_randamenty=arrow1_pozy +32+ 3*self.infoObject.current_w/96+text_height
            subtitlu_randamentx=(self.infoObject.current_w/30-10+self.infoObject.current_w / 5 + self.max_alg_size + self.max_port_size - self.infoObject.current_w / 30 + self.max_rand_size + 16-self.infoObject.current_w/30-10-text_width/2)/2

            self.display_text("RANDAMENTUL OBTINUT",subtitlu_randamentx,subtitlu_randamenty)

            text_width, text_height = self.font.size("9.9999")
            Rrx=(self.infoObject.current_w/30-10+self.infoObject.current_w / 5 + self.max_alg_size + self.max_port_size - self.infoObject.current_w / 30 + self.max_rand_size + 16-self.infoObject.current_w/30-10-text_width/2)/2
            self.display_text_green(str(round(Rr,4)),Rrx,subtitlu_randamenty+2*self.infoObject.current_w/96+text_height)

            text_width, text_height = self.font.size("RISCUL MINIM")
            risc_minimx = (self.infoObject.current_w / 30 - 10 + self.infoObject.current_w / 5 + self.max_alg_size + self.max_port_size - self.infoObject.current_w / 30 + self.max_rand_size + 16 - self.infoObject.current_w / 30 - 10 - text_width / 2) / 2
            self.display_text("RISCUL MINIM", risc_minimx,
                              subtitlu_randamenty + 4 * self.infoObject.current_w / 96 + 2*text_height)
            self.display_text_green(str(round(V[0][0], 4)), Rrx,
                              subtitlu_randamenty + 6 * self.infoObject.current_w / 96 + 3*text_height)


            if mx>arrow1_pozx and mx<arrow1_pozx+32 and my>arrow1_pozy and my<arrow1_pozy+32:
                self.screen.blit(self.left_arrow_hover,(arrow1_pozx,arrow1_pozy))
            else:
                self.screen.blit(self.left_arrow, (arrow1_pozx, arrow1_pozy))
            if mx>arrow2_pozx and mx<arrow2_pozx+32 and my>arrow2_pozy and my<arrow2_pozy+32:
                self.screen.blit(self.right_arrow_hover,(arrow2_pozx,arrow2_pozy))
            else:
                self.screen.blit(self.right_arrow, (arrow2_pozx, arrow2_pozy))


    def show_canvas(self):
        self.genereaza_randamente()
        for i in range(self.randamente_poz, self.randamente_poz + 3, 1):
            text_width, text_height = self.font.size(str(self.rmed[i]))
            if text_width > self.max_rand_size:
                self.max_rand_size = text_width
        while self.program_running:
            self.clock.tick(60)
            self.screen.fill((174, 189, 214))
            self.display_title()
            self.display_bgs()
            self.run_algorithm()
            self.display_progress()
            self.display_plot()
            mx, my = pygame.mouse.get_pos()
            self.display_play_stop(mx,my)
            self.display_values(mx,my)
            #desenare portofoliu
            j = 0
            k = 0
            for i in range(self.portofolii_poz,self.portofolii_poz+3,1):
                text_width, text_height = self.font.size(self.portofolii[i])
                if mx>self.infoObject.current_w/30 and mx<self.infoObject.current_w/30+text_width and my>self.infoObject.current_h / 20+(k+2)*self.infoObject.current_h/17 and my<self.infoObject.current_h / 20+(k+2)*self.infoObject.current_h/17+text_height:
                    self.display_text_blue(self.portofolii[i],self.infoObject.current_w/30,self.infoObject.current_h / 20+(k+2)*self.infoObject.current_h/17)
                else:
                    self.display_text(self.portofolii[i], self.infoObject.current_w/30, self.infoObject.current_h / 20 + (k + 2) * self.infoObject.current_h/17)
                if self.pozitie_portofoliu_selectat==i:
                    self.display_text_green(self.portofolii[i], self.infoObject.current_w/30, self.infoObject.current_h / 20 + (k + 2) * self.infoObject.current_h/17)
                j+=2
                k+=1
            #desenare algoritmi
            k=0
            j=0
            for i in range(self.algoritmi_poz, self.algoritmi_poz + 3, 1):
                text_width, text_height = self.font.size(self.algoritmi[i])
                if mx > self.infoObject.current_w/10 + self.max_port_size and mx < self.infoObject.current_w/10 + self.max_port_size + text_width and my > self.infoObject.current_h / 20 + (
                        k + 2) * 50 and my < self.infoObject.current_h / 20 + (k + 2) * self.infoObject.current_h/17 + text_height:
                    self.display_text_blue(self.algoritmi[i], self.infoObject.current_w/10 + self.max_port_size,
                                           self.infoObject.current_h / 20 + (k + 2) * self.infoObject.current_h/17)
                else:
                    self.display_text(self.algoritmi[i], self.infoObject.current_w/10 + self.max_port_size,
                                      self.infoObject.current_h / 20 + (k + 2) * self.infoObject.current_h/17)
                if self.pozitie_algoritm_selectat == i:
                    self.display_text_green(self.algoritmi[i], self.infoObject.current_w/10 + self.max_port_size,
                                            self.infoObject.current_h / 20 + (k + 2) * self.infoObject.current_h/17)
                j += 2
                k += 1
            #desenare randamente
            k = 0
            j = 0
            for i in range(self.randamente_poz, self.randamente_poz + 3, 1):
                text_width, text_height = self.font.size(str(self.rmed[i]))
                if mx > self.infoObject.current_w/5 + self.max_alg_size+self.max_port_size and mx < self.infoObject.current_w/5 + self.max_alg_size+self.max_port_size + text_width and my > self.infoObject.current_h / 20 + (
                        k + 2) * 50 and my < self.infoObject.current_h / 20 + (k + 2) * self.infoObject.current_h/17 + text_height:
                    self.display_text_blue(str(self.rmed[i]), self.infoObject.current_w/5 + self.max_alg_size+self.max_port_size,
                                           self.infoObject.current_h / 20 + (k + 2) * self.infoObject.current_h/17)
                else:
                    self.display_text(str(self.rmed[i]), self.infoObject.current_w/5 + self.max_alg_size+self.max_port_size,
                                      self.infoObject.current_h / 20 + (k + 2) * self.infoObject.current_h/17)
                if self.pozitie_randament_selectat == i:
                    self.display_text_green(str(self.rmed[i]), self.infoObject.current_w/5 + self.max_alg_size+self.max_port_size,
                                            self.infoObject.current_h / 20 + (k + 2) * self.infoObject.current_h/17)
                j += 2
                k += 1
            #desenare sageti
            self.draw_arrows(self.infoObject.current_w/15,self.infoObject.current_h / 20+self.infoObject.current_h/17,self.infoObject.current_h / 20 + (j-1)*self.infoObject.current_h/17,mx,my)
            self.draw_arrows(self.infoObject.current_w/7.5+self.max_port_size, self.infoObject.current_h / 20 + self.infoObject.current_h/17, self.infoObject.current_h / 20 + (j - 1) * self.infoObject.current_h/17,
                             mx, my)
            self.draw_arrows(self.infoObject.current_w/5 + self.max_alg_size+self.max_port_size, self.infoObject.current_h / 20 + self.infoObject.current_h/17,
                             self.infoObject.current_h / 20 + (j - 1) * self.infoObject.current_h/17,
                             mx, my)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.thread_running=False
                        self.program_running=False
                        self.running=False
                if event.type == pygame.QUIT:
                    self.program_running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        #modificare contor portofoliu
                        if mx>self.infoObject.current_w/15 and mx<self.infoObject.current_w/15+32 and my>self.infoObject.current_h / 20+self.infoObject.current_h/17 and my<self.infoObject.current_h / 20+self.infoObject.current_h/17+32:
                            if self.portofolii_poz>0:
                                self.portofolii_poz-=1
                        if mx>self.infoObject.current_w/15 and mx<self.infoObject.current_w/15+32 and my>self.infoObject.current_h / 20 + (j-1)*self.infoObject.current_h/17 and my<self.infoObject.current_h / 20 + (j-1)*self.infoObject.current_h/17+32:
                            if self.portofolii_poz+2<len(self.portofolii)-1:
                                self.portofolii_poz+=1
                        # modificare contor algoritm
                        if mx>self.infoObject.current_w/7.5+self.max_port_size and mx<self.infoObject.current_w/7.5+self.max_port_size+32 and my>self.infoObject.current_h / 20+self.infoObject.current_h/17 and my<self.infoObject.current_h / 20+self.infoObject.current_h/17+32:
                            if self.algoritmi_poz>0:
                                self.algoritmi_poz-=1
                        if mx>self.infoObject.current_w/7.5+self.max_port_size and mx<self.infoObject.current_w/7.5+self.max_port_size+32 and my>self.infoObject.current_h / 20 + (j-1)*self.infoObject.current_h/17 and my<self.infoObject.current_h / 20 + (j-1)*self.infoObject.current_h/17+32:
                            if self.algoritmi_poz+2<len(self.algoritmi)-1:
                                self.algoritmi_poz+=1
                        #modificare contor randament
                        if mx>self.infoObject.current_w/5 + self.max_alg_size+self.max_port_size and mx<self.infoObject.current_w/5 + self.max_alg_size+self.max_port_size+32 and my>self.infoObject.current_h / 20+self.infoObject.current_h/17 and my<self.infoObject.current_h / 20+self.infoObject.current_h/17+32:
                            if self.randamente_poz>0:
                                self.randamente_poz-=1
                        if mx>self.infoObject.current_w/5 + self.max_alg_size+self.max_port_size and mx<self.infoObject.current_w/5 + self.max_alg_size+self.max_port_size+32 and my>self.infoObject.current_h / 20 + (j-1)*self.infoObject.current_h/17 and my<self.infoObject.current_h / 20 + (j-1)*self.infoObject.current_h/17+32:
                            if self.randamente_poz+2<len(self.rmed)-1:
                                self.randamente_poz+=1
                        #modificare contor actiuni
                        arrow1_pozx = self.infoObject.current_w / 30
                        arrow1_pozy = self.infoObject.current_h / 20 + self.infoObject.current_h / 17 - 8 + 6 * self.infoObject.current_h / 17 + 63 + 15
                        arrow2_pozx = self.infoObject.current_w / 30 - 10 + self.infoObject.current_w / 5 + self.max_alg_size + self.max_port_size - self.infoObject.current_w / 30 + self.max_rand_size + 16 - 32 - 10
                        arrow2_pozy = self.infoObject.current_h / 20 + self.infoObject.current_h / 17 - 8 + 6 * self.infoObject.current_h / 17 + 63 + 15
                        if mx > arrow1_pozx and mx < arrow1_pozx + 32 and my > arrow1_pozy and my < arrow1_pozy + 32:
                            if self.actiuni_poz > 0:
                                self.actiuni_poz -= 1
                        if mx > arrow2_pozx and mx < arrow2_pozx + 32 and my > arrow2_pozy and my < arrow2_pozy + 32:
                            if self.actiuni_poz + 4 < self.nr_actiuni:
                                self.actiuni_poz += 1
                        #selectare portofoliu
                        k=0
                        for i in range(self.portofolii_poz, self.portofolii_poz + 3, 1):
                            text_width, text_height = self.font.size(self.portofolii[i])
                            if mx > self.infoObject.current_w/30 and mx < self.infoObject.current_w/30 + text_width and my > self.infoObject.current_h / 20 + (
                                    k + 2) * self.infoObject.current_h/17 and my < self.infoObject.current_h / 20 + (k + 2) * self.infoObject.current_h/17 + text_height:
                                self.portofoliu_selectat=self.portofolii[i]
                                self.pozitie_portofoliu_selectat=i
                                self.genereaza_randamente()
                                self.randamente_poz = 0

                            k += 1
                        #selectare algoritm
                        k=0
                        for i in range(self.algoritmi_poz, self.algoritmi_poz + 3, 1):
                            text_width, text_height = self.font.size(self.algoritmi[i])
                            if mx > self.infoObject.current_w/10 + self.max_port_size and mx < self.infoObject.current_w/10 + self.max_port_size + text_width and my > self.infoObject.current_h / 20 + (
                                    k + 2) * self.infoObject.current_h/17 and my < self.infoObject.current_h / 20 + (k + 2) * self.infoObject.current_h/17 + text_height:
                                self.algoritm_selectat=self.algoritmi[i]
                                self.pozitie_algoritm_selectat=i
                            k += 1
                        #selectare randament
                        k = 0
                        for i in range(self.randamente_poz, self.randamente_poz + 3, 1):
                            text_width, text_height = self.font.size(str(self.rmed[i]))
                            if mx > self.infoObject.current_w/5 + self.max_alg_size+self.max_port_size and mx < self.infoObject.current_w/5 + self.max_alg_size+self.max_port_size + text_width and my > self.infoObject.current_h / 20 + (
                                    k + 2) * self.infoObject.current_h/17 and my < self.infoObject.current_h / 20 + (k + 2) * self.infoObject.current_h/17 + text_height:
                                self.randament_selectat = self.rmed[i]
                                self.pozitie_randament_selectat = i
                            k += 1
                        if mx>self.playx and mx<self.playx+64 and my>self.playy and my<self.playy+64:
                            self.progress=0
                            if self.running==False:
                                self.running=True
                                self.initialization()
                                self.get_actiuni_portofolii()
                            else:
                                self.running=False
            pygame.display.update()


program=Program()
program.show_canvas()




