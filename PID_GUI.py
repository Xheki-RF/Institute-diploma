from tkinter import *
from RK import *
import numpy as np
from scipy.integrate import simpson
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import signal
import copy


leftCoefs = [0.0000063, 0.0071, 0.12, 0]
rightCoefs = [2]
initState = [0, 0, 0]
inputFunc = 1
step = 0.001
regTime = 2
regStructList = [1, 1, 1]

alpha = 1   # коэффициент отражения alpha 1
beta = 0.5  # коэффициент сжатия beta выбирается равным 0.5
gamma = 2   # коэффициент растяжения gamma 2
reg_default_1 = np.array([[[0.4, 0.5], [1.0, 1.2]],
                          [[1.0, 1.2], [1.5, 2.5]],
                          [[0.2, 0.25], [0.45, 0.6]]], dtype=object)
reg_default_2 = np.array([[[0.2, 0.4], [0.8, 1.0]], [[0.8, 1.0], [1.0, 1.5]], [[0.1, 0.15], [0.25, 0.25]]], dtype=object)

stepgrad, Lr = 0.01, 0.1
saver1 = copy.deepcopy(reg_default_1)
saver2 = copy.deepcopy(reg_default_2)

"""========================================WINDOW========================================"""

window = Tk()
window.title("Адаптивный нелинейный регулятор")
window.geometry("1200x900")
window["bg"] = "#D3D3D3"

text_right = StringVar()
text_left = StringVar()

current_plot = Label(window, text="Изначальный переходной процесс", width=34, height=2, bg="#FF0000")
current_plot.place(x=860, y=10)

error_text_1 = Label(window, text="g:", width=2, height=2, bg="#FF0000")
error_text_1.place(x=860, y=46)

optimized_plot = Label(window, text="Оптимизированный переходной процесс", width=34, height=2, bg="#00FF00")
optimized_plot.place(x=860, y=84)

error_text_2 = Label(window, text="g:", width=2, height=2, bg="#00FF00")
error_text_2.place(x=860, y=120)

error_label_1 = Label(window, text="", width=29, height=2, bg="#D3D3D3")
error_label_1.place(x=880, y=46)

error_label_2 = Label(window, text="", width=29, height=2, bg="#D3D3D3")
error_label_2.place(x=880, y=120)

control_object_left = Label(window, text="Коэффициенты левой части ДУ:", width=25, height=2, bg="#D3D3D3")
control_object_left.place(x=860, y=210)

co_left = Entry(window, textvariable=text_left, font="12", width=25)
co_left.place(x=860, y=250)
text_left.set("0.0000063, 0.0071, 0.12, 0")

control_object_right = Label(window, text="Коэффициенты правой части ДУ:", width=26, height=2, bg="#D3D3D3")
control_object_right.place(x=860, y=280)

co_right = Entry(window, textvariable=text_right, font="12", width=25)
co_right.place(x=860, y=320)
text_right.set("2")

non_linear_elements = Label(window, text="Подобранные параметры нелинейного элемента", width=45, height=2, bg="#C0C0C0")
non_linear_elements.place(x=20, y=500)

p_elements = Label(window, text="П канал", width=42, height=2, bg="#C0C0C0")
p_elements.place(x=20, y=550)

i_elements = Label(window, text="И канал", width=42, height=2, bg="#C0C0C0")
i_elements.place(x=350, y=550)

d_elements = Label(window, text="Д канал", width=42, height=2, bg="#C0C0C0")
d_elements.place(x=680, y=550)

fig_p = Figure(figsize=[3, 3])
canvas_p = FigureCanvasTkAgg(fig_p, window)
canvas_p.draw()
canvas_p.get_tk_widget().place(x=20, y=590)

fig_i = Figure(figsize=[3, 3])
canvas_i = FigureCanvasTkAgg(fig_i, window)
canvas_i.draw()
canvas_i.get_tk_widget().place(x=350, y=590)

fig_d = Figure(figsize=[3, 3])
canvas_d = FigureCanvasTkAgg(fig_d, window)
canvas_d.draw()
canvas_d.get_tk_widget().place(x=680, y=590)


def connection(regulator):
    reg_p_x = [0, regulator[0][0][0], regulator[0][1][0], regulator[0][1][0] + 1]
    reg_p_y = [0, regulator[0][0][1], regulator[0][1][1], regulator[0][1][1]]

    reg_i_x = [0, regulator[1][0][0], regulator[1][1][0], regulator[1][1][0] + 1]
    reg_i_y = [0, regulator[1][0][1], regulator[1][1][1], regulator[1][1][1]]

    reg_d_x = [0, regulator[2][0][0], regulator[2][1][0], regulator[2][1][0] + 1]
    reg_d_y = [0, regulator[2][0][1], regulator[2][1][1], regulator[2][1][1]]

    p_reg_plot = [reg_p_x, reg_p_y]
    i_reg_plot = [reg_i_x, reg_i_y]
    d_reg_plot = [reg_d_x, reg_d_y]

    return p_reg_plot, i_reg_plot, d_reg_plot


def get_value():
    global epochs
    epochs = int(entry.get())

    temp = co_right.get()
    rightCoefs1 = []
    temp = temp.split(", ")

    for i in range(len(temp)):
        rightCoefs1.append(float(temp[i]))

    temp2 = co_left.get()
    leftCoefs1 = []
    temp2 = temp2.split(", ")

    for i in range(len(temp2)):
        leftCoefs1.append(float(temp2[i]))

    global params
    global params2
    params = [leftCoefs1, rightCoefs1, initState, inputFunc, step, regTime, reg_default_1, reg_default_2, stepgrad, Lr,
              epochs]
    params2 = [leftCoefs1, rightCoefs1, initState, inputFunc, step, regTime, saver1, saver2, stepgrad, Lr, epochs]


fig = Figure()
canvas = FigureCanvasTkAgg(fig, window)
canvas.draw()
canvas.get_tk_widget().place(x=200, y=0)

f_top = Label(text="Методы:", width=15, height=2, bg="#D3D3D3")
f_top.place(x=20, y=10)

epoch_num = Label(text="Введите количество эпох:", width=20, height=2, bg="#D3D3D3")
epoch_num.place(x=10, y=340)

entry = Entry(window, font="12", width=14)
entry.place(x=22, y=380)

button = Button(window, text="Enter", command=get_value, width=12, height=2, bg="#C0C0C0")
button.place(x=37, y=405)


def simps(y, step):
    return simpson(y, dx=step)


"""========================================NELDER-MID========================================"""


def Nelder_Mid_dots():
    func_list = list()

    kp1 = reg_default_1[0]
    ki1 = reg_default_1[1]
    kd1 = reg_default_1[2]

    ap, p_aelta = kp1[0][0], 0.1
    bp, p_belta = kp1[0][1], 0.1
    cp, p_celta = kp1[1][0], 0.1
    dp, p_delta = kp1[1][1], 0.12

    ai, i_aelta = ki1[0][0], 0.1
    bi, i_belta = ki1[0][1], 0.1
    ci, i_celta = ki1[1][0], 0.1
    di, i_delta = ki1[1][1], 0.12

    ad, d_aelta = kd1[0][0], 0.1
    bd, d_belta = kd1[0][1], 0.1
    cd, d_celta = kd1[1][0], 0.1
    dd, d_delta = kd1[1][1], 0.12

    p_list = [np.array(
        [[abs(np.random.uniform(ap - p_aelta, ap + p_aelta)), abs(np.random.uniform(bp - p_belta, bp + p_belta))],
         [abs(np.random.uniform(cp - p_celta, cp + p_celta)), abs(np.random.uniform(dp - p_delta, dp + p_delta))]]) for
        i in range(4)]

    i_list = [np.array(
        [[abs(np.random.uniform(ai - i_aelta, ai + i_aelta)), abs(np.random.uniform(bi - i_belta, bi + i_belta))],
         [abs(np.random.uniform(ci - i_celta, ci + i_celta)), abs(np.random.uniform(di - i_delta, di + i_delta))]]) for
        i in range(4)]

    d_list = [np.array(
        [[abs(np.random.uniform(ad - d_aelta, ad + d_aelta)), abs(np.random.uniform(bd - d_belta, bd + d_belta))],
         [abs(np.random.uniform(cd - d_celta, cd + d_celta)), abs(np.random.uniform(dd - d_delta, dd + d_delta))]]) for
        i in range(4)]

    r = [reg_default_1]

    for i in range(4):
        r.append([p_list[i], i_list[i], d_list[i]])

    for el, el1, el2 in r:
        my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                            el, el1, el2, Tr=regTime / 5)
        my_RK.solve()
        y = my_RK.e0_solve
        e = simpson(abs(np.array(y)), dx=step)
        func_list.append(e)

    return r, func_list


def Nelder_Mid_method(k_list, func_list):
    r = []
    key_val = list(zip(k_list, func_list))
    key_val = sorted(key_val, key=lambda tup: tup[1])

    kp_list = [x[0][0] for x in key_val]
    ki_list = [x[0][1] for x in key_val]
    kd_list = [x[0][2] for x in key_val]

    kp_center_mass = sum(kp_list[:-1]) / len(kp_list[:-1])
    ki_center_mass = sum(ki_list[:-1]) / len(ki_list[:-1])
    kd_center_mass = sum(kd_list[:-1]) / len(kd_list[:-1])

    kp_r = (1 + alpha) * kp_center_mass - alpha * kp_list[4]
    ki_r = (1 + alpha) * ki_center_mass - alpha * ki_list[4]
    kd_r = (1 + alpha) * kd_center_mass - alpha * kd_list[4]

    my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                        kp_r, ki_r, kd_r, Tr=regTime / 5)
    my_RK.solve()
    y = my_RK.e0_solve
    e_r = simpson(abs(np.array(y)), dx=step)

    if e_r < func_list[0]:

        kp_e = (1 - gamma) * kp_center_mass + gamma * kp_r
        ki_e = (1 - gamma) * ki_center_mass + gamma * ki_r
        kd_e = (1 - gamma) * kd_center_mass + gamma * kd_r

        my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                            kp_e, ki_e, kd_e, Tr=regTime / 5)
        my_RK.solve()
        y = my_RK.e0_solve
        e_e = simpson(abs(np.array(y)), dx=step)

        if e_e < e_r:

            kp_list[4], func_list[4] = kp_e, e_e
            ki_list[4], func_list[4] = ki_e, e_e
            kd_list[4], func_list[4] = kd_e, e_e

            r = []
            for i in range(5):
                r.append([kp_list[i], ki_list[i], kd_list[i]])
            return r, func_list

        else:
            kp_list[4], func_list[4] = kp_r, e_r
            ki_list[4], func_list[4] = ki_r, e_r
            kd_list[4], func_list[4] = kd_r, e_r

            for i in range(5):
                r.append([kp_list[i], ki_list[i], kd_list[i]])
            return r, func_list

    elif func_list[0] < e_r < func_list[3]:

        kp_list[4], func_list[4] = kp_r, e_r
        ki_list[4], func_list[4] = ki_r, e_r
        kd_list[4], func_list[4] = kd_r, e_r

        for i in range(5):
            r.append([kp_list[i], ki_list[i], kd_list[i]])
        return r, func_list

    elif func_list[3] < e_r < func_list[4]:

        kp_list[4], func_list[4] = kp_r, e_r
        ki_list[4], func_list[4] = ki_r, e_r
        kd_list[4], func_list[4] = kd_r, e_r

    kp_s = beta * kp_list[4] + (1 - beta) * kp_center_mass
    ki_s = beta * ki_list[4] + (1 - beta) * ki_center_mass
    kd_s = beta * kd_list[4] + (1 - beta) * kd_center_mass

    my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                        kp_s, ki_s, kd_s, Tr=regTime / 5)
    my_RK.solve()
    y = my_RK.e0_solve
    e_s = simpson(abs(np.array(y)), dx=step)

    if e_s < func_list[4]:
        kp_list[4], func_list[4] = kp_s, e_s
        ki_list[4], func_list[4] = ki_s, e_s
        kd_list[4], func_list[4] = kd_s, e_s
        r = []
        for i in range(5):
            r.append([kp_list[i], ki_list[i], kd_list[i]])
        return r, func_list
    else:
        for k in range(4):
            if k == 3:
                continue
            else:
                kp_list[k] = (kp_list[3] + (kp_list[k] - kp_list[3])) / 2
                ki_list[k] = (ki_list[3] + (ki_list[k] - ki_list[3])) / 2
                kd_list[k] = (kd_list[3] + (kd_list[k] - kd_list[3])) / 2

                my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                                    kp_list[k], ki_list[k], kd_list[k], Tr=regTime / 5)
                my_RK.solve()
                y = my_RK.e0_solve
                e = simpson(abs(np.array(y)), dx=step)
                func_list[k] = e

                r = []
                for i in range(5):
                    r.append([kp_list[i], ki_list[i], kd_list[i]])
        return r, func_list


def Nelder_Mid(k, func_list):
    print('\n')
    i = 0
    while i != epochs:
        k, func_list = Nelder_Mid_method(k, func_list)

        print('-' * 50)
        print('epoch: ', i + 1)
        print('-' * 50)
        i += 1
    return k[func_list.index(min(func_list))]


"""========================================ADAPTIVE GRADIENT========================================"""


def abs_stability_criteria(regStructList, reg, Hp, Hi, Hd, tolerance):
    if regStructList[0] == 1:
        K1p = reg[0][0][1] / reg[0][0][0]
        K2p = (reg[0][1][1] - reg[0][0][1]) / (reg[0][1][0] - reg[0][0][0])
        KpMax = max(K1p, K2p)
    else:
        KpMax = 0
    if regStructList[1] == 1:
        K1i = reg[1][0][1] / reg[1][0][0]
        K2i = (reg[1][1][1] - reg[1][0][1]) / (reg[1][1][0] - reg[1][0][0])
        KiMax = max(K1i, K2i)
    else:
        KiMax = 0
    if regStructList[2] == 1:
        K1d = reg[2][0][1] / reg[2][0][0]
        K2d = (reg[2][1][1] - reg[2][0][1]) / (reg[2][1][0] - reg[2][0][0])
        KdMax = max(K1d, K2d)
    else:
        KdMax = 0
    for i in range(len(Hp.real)):
        tmp = np.matrix([[1 + KpMax * Hp.real[i], KpMax * Hp.real[i], KpMax * Hp.real[i]],
                         [KiMax * Hi.real[i], 1 + KiMax * Hi.real[i], KiMax * Hi.real[i]],
                         [KdMax * Hd.real[i], KdMax * Hd.real[i], 1 + KdMax * Hd.real[i]]])
        res = np.linalg.det(tmp)
        if abs(res) <= tolerance:
            return -1
    return 0


def func(leftCoefs, rightCoefs, inputFunc, reg2):
    timeConst = 0.5
    satLevel = reg2[0][1][1]
    plantTF = signal.TransferFunction(rightCoefs, leftCoefs)
    wp, Hp = signal.freqresp(plantTF)
    tmp = leftCoefs[:]
    tmp.extend([0, ])
    plantIntgrTF = signal.TransferFunction(rightCoefs, tmp)
    wi, Hi = signal.freqresp(plantIntgrTF)
    tmp = rightCoefs[:]
    tmp.extend([0, ])
    plantDrvtTF = signal.TransferFunction(tmp, leftCoefs)
    wd, Hd = signal.freqresp(plantDrvtTF)

    regsList = []

    dList = [0, 0, 0]
    cList = [0, 0, 0]
    bList = [0, 0, 0]
    aList = [0, 0, 0]

    if leftCoefs[-1] != 0:
        if regStructList[1] == 1:
            dList[1] = random.uniform(inputFunc * leftCoefs[-1] / (satLevel * rightCoefs[-1]),
                                      2 * inputFunc * leftCoefs[-1] / (satLevel * rightCoefs[-1]))
            temp = 1 - dList[1]
        if regStructList[0] == 1:
            dList[0] = random.uniform(0, temp)
        if regStructList[2] == 1:
            dList[2] = random.uniform(0, 0.1)
    else:
        if regStructList[0] == 1:
            dList[0] = random.uniform(0.5, 1)
        if regStructList[1] == 1:
            dList[1] = random.uniform(0, 0.4)
        if regStructList[2] == 1:
            dList[2] = random.uniform(0, 0.1)

    # Normalize this values to have sum of saturation
    sumDList = sum(dList)
    for i in range(len(dList)):
        dList[i] = dList[i] * satLevel / sumDList

    for i in range(len(regStructList)):
        if regStructList[i] == 1:
            bList[i] = random.uniform(0, dList[i])

    if regStructList[0] == 1:
        cList[0] = random.uniform(inputFunc, 2 * inputFunc)
    if regStructList[1] == 1:
        cList[1] = random.uniform(inputFunc * timeConst, 5 * inputFunc * timeConst)
    if regStructList[2] == 1:
        cList[2] = random.uniform(inputFunc / timeConst, 5 * inputFunc / timeConst)

    if regStructList[0] == 1:
        aList[0] = random.uniform(0, inputFunc)
    if regStructList[1] == 1:
        aList[1] = random.uniform(0, inputFunc * timeConst)
    if regStructList[2] == 1:
        aList[2] = random.uniform(0, inputFunc / timeConst)

    # Check if Yackubovich condition is met
    tmpReg = []
    if regStructList[0] == 1:
        tmpReg.append([[aList[0], bList[0]], [cList[0], dList[0]]])
    else:
        tmpReg.append(None)
    if regStructList[1] == 1:
        tmpReg.append([[aList[1], bList[1]], [cList[1], dList[1]]])
    else:
        tmpReg.append(None)
    if regStructList[2] == 1:
        tmpReg.append([[aList[2], bList[2]], [cList[2], dList[2]]])
    else:
        tmpReg.append(None)

    res = abs_stability_criteria(regStructList, reg2, Hp, Hi, Hd, 0.001)

    return res


def gradient(kp_1, kp_2, e_1, e_2):
    gradient_vector = np.zeros((2, 2))
    for j in range(2):
        for i in range(2):
            a = (kp_1[j][i] * (e_1 - e_2)) / (kp_1[j][i] - kp_2[j][i])
            gradient_vector[j][i] = a
    return gradient_vector


def gradient_descend(*params):
    reg1 = copy.deepcopy(params[6])
    reg2 = copy.deepcopy(params[7])

    step_grad = params[8]

    e_lst = []
    f_lst_p = []
    f_lst_i = []
    f_lst_d = []

    kp_1 = reg1[0]
    kp_2 = reg2[0]

    ki_1 = reg1[1]
    ki_2 = reg2[1]

    kd_1 = reg1[2]
    kd_2 = reg2[2]

    print('-' * 50)
    lr = Lr

    for i in range(epochs):

        my_RK_1 = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                              kp_1, ki_1, kd_1, Tr=regTime / 5)
        my_RK_2 = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                              kp_2, ki_2, kd_2, Tr=regTime / 5)

        my_RK_1.solve()
        my_RK_2.solve()

        e_1 = simps(abs(np.asarray(my_RK_1.e0_solve)), step)  # получаем значения ошибки
        e_2 = simps(abs(np.asarray(my_RK_2.e0_solve)), step)  # получаем значения ошибки

        gradient_vector1 = np.asarray(gradient(kp_1, kp_2, e_1, e_2))
        gradient_vector2 = np.asarray(gradient(ki_1, ki_2, e_1, e_2))
        gradient_vector3 = np.asarray(gradient(kd_1, kd_2, e_1, e_2))  # Поучили вектор производных по каждой компоненте

        kp_3 = kp_2 - lr * gradient_vector1
        ki_3 = ki_2 - lr * gradient_vector2
        kd_3 = kd_2 - lr * gradient_vector3

        reg2[0] = kp_2 - lr * gradient_vector1
        reg2[1] = ki_2 - lr * gradient_vector2
        reg2[2] = kd_2 - lr * gradient_vector3

        if func(leftCoefs, rightCoefs, inputFunc, reg2) == 0:

            kp_1 = kp_2.copy()
            ki_1 = ki_2.copy()
            kd_1 = kd_2.copy()

            reg1[0] = kp_1
            reg1[1] = ki_1
            reg1[2] = kd_1

            kp_2 = kp_2 - lr * gradient_vector1
            ki_2 = ki_2 - lr * gradient_vector2
            kd_2 = kd_2 - lr * gradient_vector3

            reg2[0] = kp_2
            reg2[1] = ki_2
            reg2[2] = kd_2

            lr = Lr

            print('Absolutely stable:',
                  func(leftCoefs, rightCoefs, inputFunc, reg2))

            my_RK_3 = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                                  kp_2, ki_2, kd_2, Tr=regTime / 5)
            my_RK_3.solve()
            e = simps(abs(np.asarray(my_RK_3.e0_solve)), step)
            print('Error: ', e, '\n', 'step = ', i + 1)
            print('-' * 50)
            e_lst.append(e)
            f_lst_p.append(kp_2)
            f_lst_i.append(ki_2)
            f_lst_d.append(kd_2)

        else:
            lr -= step_grad
            reg2[0] = kp_2
            reg2[1] = ki_2
            reg2[2] = kd_2
            print('Absolutely unstable')
            print('step:', i + 1, '\n')
            print('-' * 50)

    reg2 = [f_lst_p[e_lst.index(min(e_lst))], f_lst_i[e_lst.index(min(e_lst))], f_lst_d[e_lst.index(min(e_lst))]]
    print(reg2)

    return reg2


"""========================================NORMAL GRADIENT========================================"""


def gradient_descend_normal(*params):
    error_list = []
    reg = params[6]
    reg2 = params[7]

    kp_1, kp_2 = reg[0], reg2[0]
    ki_1, ki_2 = reg[1], reg2[1]
    kd_1, kd_2 = reg[2], reg2[2]

    non_linear_element_dots_p = []
    non_linear_element_dots_i = []
    non_linear_element_dots_d = []
    reg_x = [[], [], []]

    for i in range(epochs):
        my_RK_1 = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                              kp_1, ki_1, kd_1, Tr=regTime / 5)
        my_RK_2 = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                              kp_2, ki_2, kd_2, Tr=regTime / 5)

        my_RK_1.solve()
        my_RK_2.solve()

        e_1 = simps(abs(np.asarray(my_RK_1.e0_solve)), step)
        e_2 = simps(abs(np.asarray(my_RK_2.e0_solve)), step)

        gradient_vector_p = np.asarray(gradient(kp_1, kp_2, e_1, e_2))
        gradient_vector_i = np.asarray(gradient(ki_1, ki_2, e_1, e_2))
        gradient_vector_d = np.asarray(gradient(kd_1, kd_2, e_1, e_2))

        kp_1 = kp_2
        kp_2 = kp_2 - Lr * gradient_vector_p

        ki_1 = ki_2
        ki_2 = ki_2 - Lr * gradient_vector_i

        kd_1 = kd_2
        kd_2 = kd_2 - Lr * gradient_vector_d

        reg_x[0] = kp_2 - Lr * gradient_vector_p
        reg_x[1] = ki_2 - Lr * gradient_vector_i
        reg_x[2] = kd_2 - Lr * gradient_vector_d

        if func(leftCoefs, rightCoefs, inputFunc, reg_x) == 0:

            non_linear_element_dots_p.append(kp_2)
            non_linear_element_dots_i.append(ki_2)
            non_linear_element_dots_d.append(kd_2)

            my_RK_3 = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                                  kp_2, ki_2, kd_2, Tr=regTime / 5)
            my_RK_3.solve()
            e = simps(abs(np.asarray(my_RK_3.e0_solve)), step)
            error_list.append(e)
            print('Error: ', e, '\n', 'step = ', i + 1, '\n')
            print('-' * 50)
        else:
            pass
    # print("Минимальная ошибка: ", min(error_list))
    print("Оптимальные параметры нелинейного элемента:", "\n", "П канал:", non_linear_element_dots_p[error_list.index(min(error_list))], "\n",
    "И канал:", non_linear_element_dots_i[error_list.index(min(error_list))], "\n",
    "Д канал:", non_linear_element_dots_d[error_list.index(min(error_list))])

    # return kp_2, ki_2, kd_2
    return non_linear_element_dots_p[error_list.index(min(error_list))], non_linear_element_dots_i[error_list.index(min(error_list))], non_linear_element_dots_d[error_list.index(min(error_list))]


"""========================================GUI========================================"""


def submit_nelder_mid(*params):
    global canvas
    global canvas_p, canvas_i, canvas_d
    leftCoefs = params[0]
    rightCoefs = params[1]
    reg_nd = [[], [], []]
    reg_nd[0] = saver1[0].copy()
    reg_nd[1] = saver1[1].copy()
    reg_nd[2] = saver1[2].copy()

    fig = Figure()
    plot1 = fig.add_subplot(111)
    my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                        reg_nd[0], reg_nd[1], reg_nd[2], Tr=regTime / 5)
    my_RK.solve()
    y = np.asarray(my_RK.z0_solve)
    error = simps(abs(np.asarray(my_RK.e0_solve)), step)

    plot1.plot(y, color="#FF0000")

    k_nd, func_list_nd = Nelder_Mid_dots()
    reg_nd_new = Nelder_Mid(k_nd, func_list_nd)
    print(reg_nd_new)
    my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                        reg_nd_new[0], reg_nd_new[1], reg_nd_new[2], Tr=regTime / 5)
    my_RK.solve()
    y = np.asarray(my_RK.z0_solve)

    error_new = simps(abs(np.asarray(my_RK.e0_solve)), step)
    if func(leftCoefs, rightCoefs, inputFunc, reg_nd_new) == 0:
        print("Stable")
    else:
        print("Unstable")

    plot1.plot(y, color="#00FF00")

    plot1.legend(["Изначальный", "Оптимизированный"], loc="lower right")
    plot1.set_xlabel("мс")
    plot1.set_title("Переходные процессы")
    plot1.grid()
    canvas = FigureCanvasTkAgg(fig, window)
    canvas.draw()
    canvas.get_tk_widget().place(x=200, y=0)

    error_label_1.config(text=error, width=29, height=2)
    error_label_2.config(text=error_new, width=29, height=2)

    p_nelder, i_nelder, d_nelder = connection(reg_nd_new)

    fig_p = Figure(figsize=[3, 3])
    p_nelder_plot = fig_p.add_subplot(111)
    p_nelder_plot.plot(p_nelder[0], p_nelder[1], "#FF0000")
    p_nelder_plot.plot(p_nelder[0], p_nelder[1], "r.")
    p_nelder_plot.grid()
    canvas_p = FigureCanvasTkAgg(fig_p, window)
    canvas_p.draw()
    canvas_p.get_tk_widget().place(x=20, y=590)

    fig_i = Figure(figsize=[3, 3])
    i_nelder_plot = fig_i.add_subplot(111)
    i_nelder_plot.plot(i_nelder[0], i_nelder[1], "#00FF00")
    i_nelder_plot.plot(i_nelder[0], i_nelder[1], "g.")
    i_nelder_plot.grid()
    canvas_i = FigureCanvasTkAgg(fig_i, window)
    canvas_i.draw()
    canvas_i.get_tk_widget().place(x=350, y=590)

    fig_d = Figure(figsize=[3, 3])
    d_nelder_plot = fig_d.add_subplot(111)
    d_nelder_plot.plot(d_nelder[0], d_nelder[1], "#0000FF")
    d_nelder_plot.plot(d_nelder[0], d_nelder[1], "b.")
    d_nelder_plot.grid()
    canvas_d = FigureCanvasTkAgg(fig_d, window)
    canvas_d.draw()
    canvas_d.get_tk_widget().place(x=680, y=590)


def submit_adaptive_gradient(*params):
    global canvas
    global canvas_p, canvas_i, canvas_d
    leftCoefs = params[0]
    rightCoefs = params[1]
    reg_adp_1 = [[], [], []]
    reg_adp_1[0] = saver1[0].copy()
    reg_adp_1[1] = saver1[1].copy()
    reg_adp_1[2] = saver1[2].copy()

    fig = Figure()
    plot1 = fig.add_subplot(111)
    my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                        reg_adp_1[0], reg_adp_1[1], reg_adp_1[2], Tr=regTime / 5)
    my_RK.solve()
    y = np.asarray(my_RK.z0_solve)
    error = simps(abs(np.asarray(my_RK.e0_solve)), step)

    plot1.plot(y, color="#FF0000")

    reg_adp = gradient_descend(*params)
    my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                        reg_adp[0], reg_adp[1], reg_adp[2], Tr=regTime / 5)
    my_RK.solve()
    y = np.asarray(my_RK.z0_solve)
    error_new = simps(abs(np.asarray(my_RK.e0_solve)), step)

    plot1.plot(y, color="#00FF00")

    plot1.legend(["Изначальный", "Оптимизированный"], loc="lower right")
    plot1.set_xlabel("мс")
    plot1.set_title("Переходные процессы")
    plot1.grid()
    canvas = FigureCanvasTkAgg(fig, window)
    canvas.draw()
    canvas.get_tk_widget().place(x=200, y=0)

    error_label_1.config(text=error, width=29, height=2)
    error_label_2.config(text=error_new, width=29, height=2)

    p_adp, i_adp, d_adp = connection(reg_adp)

    fig_p = Figure(figsize=[3, 3])
    p_adp_plot = fig_p.add_subplot(111)
    p_adp_plot.plot(p_adp[0], p_adp[1], "#FF0000")
    p_adp_plot.plot(p_adp[0], p_adp[1], "r.")
    p_adp_plot.grid()
    canvas_p = FigureCanvasTkAgg(fig_p, window)
    canvas_p.draw()
    canvas_p.get_tk_widget().place(x=20, y=590)

    fig_i = Figure(figsize=[3, 3])
    i_adp_plot = fig_i.add_subplot(111)
    i_adp_plot.plot(i_adp[0], i_adp[1], "#00FF00")
    i_adp_plot.plot(i_adp[0], i_adp[1], "g.")
    i_adp_plot.grid()
    canvas_i = FigureCanvasTkAgg(fig_i, window)
    canvas_i.draw()
    canvas_i.get_tk_widget().place(x=350, y=590)

    fig_d = Figure(figsize=[3, 3])
    d_adp_plot = fig_d.add_subplot(111)
    d_adp_plot.plot(d_adp[0], d_adp[1], "#0000FF")
    d_adp_plot.plot(d_adp[0], d_adp[1], "b.")
    d_adp_plot.grid()
    canvas_d = FigureCanvasTkAgg(fig_d, window)
    canvas_d.draw()
    canvas_d.get_tk_widget().place(x=680, y=590)


def submit_gradient(*params):
    global canvas
    global canvas_p, canvas_i, canvas_d
    leftCoefs = params[0]
    rightCoefs = params[1]
    reg_norm_1 = [[], [], []]
    reg_norm_1[0] = saver1[0].copy()
    reg_norm_1[1] = saver1[1].copy()
    reg_norm_1[2] = saver1[2].copy()

    fig = Figure()
    plot1 = fig.add_subplot(111)
    my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime,
                        reg_norm_1[0], reg_norm_1[1], reg_norm_1[2], Tr=regTime / 5)
    my_RK.solve()
    y = np.asarray(my_RK.z0_solve)
    error = simps(abs(np.asarray(my_RK.e0_solve)), step)

    plot1.plot(y, color="#FF0000")

    reg_normal = [[], [], []]
    reg_normal[0], reg_normal[1], reg_normal[2] = gradient_descend_normal(*params)
    my_RK = Runge_Kutta(leftCoefs, rightCoefs, initState, inputFunc, step, 1.5 * regTime, reg_normal[0], reg_normal[1],
                        reg_normal[2],
                        Tr=regTime / 5)
    my_RK.solve()
    y = np.asarray(my_RK.z0_solve)

    error_new = simps(abs(np.asarray(my_RK.e0_solve)), step)

    plot1.plot(y, color="#00FF00")

    plot1.legend(["Изначальный", "Оптимизированный"], loc="lower right")
    plot1.set_xlabel("мс")
    plot1.set_title("Переходные процессы")
    plot1.grid()
    canvas = FigureCanvasTkAgg(fig, window)
    canvas.draw()
    canvas.get_tk_widget().place(x=200, y=0)

    error_label_1.config(text=error, width=29, height=2)
    error_label_2.config(text=error_new, width=29, height=2)

    p_grad, i_grad, d_grad = connection(reg_normal)
    print(p_grad, i_grad, d_grad)

    fig_p = Figure(figsize=[3, 3])
    p_grad_plot = fig_p.add_subplot(111)
    p_grad_plot.plot(p_grad[0], p_grad[1], "#FF0000")
    p_grad_plot.plot(p_grad[0], p_grad[1], "r.")
    p_grad_plot.grid()
    canvas_p = FigureCanvasTkAgg(fig_p, window)
    canvas_p.draw()
    canvas_p.get_tk_widget().place(x=20, y=590)

    fig_i = Figure(figsize=[3, 3])
    i_grad_plot = fig_i.add_subplot(111)
    i_grad_plot.plot(i_grad[0], i_grad[1], "#00FF00")
    i_grad_plot.plot(i_grad[0], i_grad[1], "g.")
    i_grad_plot.grid()
    canvas_i = FigureCanvasTkAgg(fig_i, window)
    canvas_i.draw()
    canvas_i.get_tk_widget().place(x=350, y=590)

    fig_d = Figure(figsize=[3, 3])
    d_grad_plot = fig_d.add_subplot(111)
    d_grad_plot.plot(d_grad[0], d_grad[1], "#0000FF")
    d_grad_plot.plot(d_grad[0], d_grad[1], "b.")
    d_grad_plot.grid()
    canvas_d = FigureCanvasTkAgg(fig_d, window)
    canvas_d.draw()
    canvas_d.get_tk_widget().place(x=680, y=590)


def clear():
    canvas.get_tk_widget().destroy()

    error_label_1.config(text="")
    error_label_2.config(text="")

    canvas_p.get_tk_widget().destroy()
    canvas_i.get_tk_widget().destroy()
    canvas_d.get_tk_widget().destroy()


submit1 = Button(window, text="Нелдер-Мид", command=lambda: [clear(), submit_nelder_mid(*params2)], width=17, height=5,
                 bg="#C0C0C0")
submit1.place(x=20, y=50)

submit2 = Button(window, text="Адаптивный градиент", command=lambda: [clear(), submit_adaptive_gradient(*params2)],
                 width=17, height=5, bg="#C0C0C0")
submit2.place(x=20, y=140)

submit3 = Button(window, text="Градиент", command=lambda: [clear(), submit_gradient(*params2)], width=17, height=5,
                 bg="#C0C0C0")
submit3.place(x=20, y=230)

clear_but = Button(window, text="Очистить поля", command=clear, width=12, height=2, bg="#C0C0C0")
clear_but.place(x=940, y=160)

window.mainloop()