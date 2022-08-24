import tqdm
from utils import *
from itertools import groupby
from operator import itemgetter
from sklearn.linear_model import LinearRegression


# check if the points to be used for linear regression are consecutive
def is_consecutive(points):
    if all(np.diff(points) == 1):
        return True
    else:
        return False


# isolate the largest sequence of consecutive points to be used in the linear regression model
def fix_consecutive_error(points):
    length = 0
    corrected_points = []
    for k, g in groupby(enumerate(points), lambda d: d[0] - d[1]):
        temp = list(map(itemgetter(1), g))
        if len(temp) > length:
            corrected_points = temp
            length = len(temp)
    return corrected_points


# returns the points corresponding to systolic and diastolic phases.
# These points will be used to isolate the region of LA strain used to fit the linear regression model
def determine_points(strain_curve, time_axis):
    time_axis = np.array(time_axis)
    end_of_systole = time_axis[np.argmax(strain_curve) - 1]
    end_of_diastole = 0.65
    if end_of_systole > 0.5:
        end_of_systole = 0.5
        end_of_diastole = 0.8

    gradient = np.diff(strain_curve)
    positive_grad_positions = np.where(gradient >= 0.05)[0]
    negative_grad_positions = np.where(gradient <= -0.05)[0]

    # take only position that correspond to the initial 35% of the heart cycle
    # which is approximately the systolic phase
    positive_grad_positions = positive_grad_positions[time_axis[positive_grad_positions] >= 0]
    positive_grad_positions = positive_grad_positions[time_axis[positive_grad_positions] <= end_of_systole]

    if not is_consecutive(positive_grad_positions):
        positive_grad_positions = fix_consecutive_error(positive_grad_positions)

    # take only position that correspond between 40% and 80% of the heart cycle
    # which is approximately the early diastolic phase
    negative_grad_positions = negative_grad_positions[time_axis[negative_grad_positions] > end_of_systole]
    negative_grad_positions = negative_grad_positions[time_axis[negative_grad_positions] <= end_of_diastole]

    if not is_consecutive(negative_grad_positions):
        negative_grad_positions = fix_consecutive_error(negative_grad_positions)

    # return the starting and ending points during systolic and diastolic phase to calculate the slope
    systolic_phase_pos = (positive_grad_positions[0], positive_grad_positions[-1])
    diastolic_phase_pos = (negative_grad_positions[0], negative_grad_positions[-1])
    return systolic_phase_pos, diastolic_phase_pos


# fit the linear regression model and produce images to visually evaluate the fitting
def calculate_slopes(fit_data, time, ids, path):

    if not os.path.exists(os.path.join(path, "Slopes")):
        os.makedirs(os.path.join(path, "Slopes"))

    systolic_slope_lr = []
    diastolic_slope_lr = []
    for i in tqdm.tqdm(range(len(fit_data)), total=len(fit_data)):
        sys, dia = determine_points(fit_data[i], time)

        # calculate slope fitting a linear regression model
        systolic_phase_model = LinearRegression()
        systolic_phase_model.fit(time[sys[0]:sys[1]].reshape(-1, 1),
                                 fit_data[i][sys[0]:sys[1]].reshape(-1, 1))

        diastolic_phase_model = LinearRegression()
        diastolic_phase_model.fit(time[dia[0]:dia[1]].reshape(-1, 1),
                                  fit_data[i][dia[0]:dia[1]].reshape(-1, 1))

        systolic_slope_lr.append(systolic_phase_model.coef_)
        diastolic_slope_lr.append(diastolic_phase_model.coef_)

        # plot the fitted line along with the original strain curves
        plt.figure()
        plt.plot(time, fit_data[i], c="blue", alpha=0.8)
        plt.plot(time[dia[0]:dia[1]], diastolic_phase_model.predict(time[dia[0]:dia[1]].reshape(-1, 1)), c="red")
        plt.plot(time[sys[0]:sys[1]], systolic_phase_model.predict(time[sys[0]:sys[1]].reshape(-1, 1)), c="red")
        plt.xlabel("Time (% Cycle)")
        plt.ylabel("Strain (%)")
        plt.title("Fitted Linear Regression Model")
        plt.savefig(os.path.join(path, "Slopes", "Patient " + ids[i] + ".png"))
        plt.close()
    return systolic_slope_lr, diastolic_slope_lr
