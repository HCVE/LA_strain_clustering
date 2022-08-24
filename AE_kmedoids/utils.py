import os
import plotly
import xlsxwriter
import align_ecg
import upsampling as us
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from tensorflow import keras
from matplotlib.patches import Ellipse
from sklearn.linear_model import LinearRegression


def extract_latent_dim(model, fit_data):
    output1 = 'z_mean'
    z_mean = keras.Model(inputs=model.input, outputs=model.get_layer(output1).output)
    z = z_mean.predict(fit_data)
    return z


def analyze_patient(label, patient):
    ids = [None] * len(np.unique(label))
    patient = np.array(patient)
    for i in range(len(np.unique(label))):
        ids[i] = patient[np.where(label == np.unique(label)[i])]
    return ids


def write2excel(label, patient, path):
    cluster_labels = {0: 1, 1: 2, 2: 4, 3: 3, 4: 5}
    workbook = xlsxwriter.Workbook(path + 'Clustering_assignments.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, 'Patient ID')
    worksheet.write(0, 1, 'Cluster')
    for i in range(len(label)):
        worksheet.write(i + 1, 0, patient[i])
        worksheet.write(i + 1, 1, cluster_labels[label[i]])
    workbook.close()


def exclude_patients(discarded_patients1, discarded_patients2, whole_data, cut_data, patient_names, time_interval):
    # delete the data for the excluded patient
    deleted_patients = np.unique(np.concatenate((discarded_patients1, discarded_patients2)))
    deleted_patients = list(map(str, deleted_patients))
    indexes = []

    for dp in deleted_patients:
        try:
            indexes.append(patient_names.index(dp))
        except ValueError:
            print('Patient with ID: {} is not in dataset'.format(dp))

    for ind in sorted(indexes, reverse=True):
        del whole_data[ind]
        del cut_data[ind]
        del patient_names[ind]
        del time_interval[ind]

    return whole_data, cut_data, patient_names, time_interval


def get_aligned_signals(extracted_data, select_point, interval, patient_id, reference_id, avc_times, p_wave_times):
    # normalize the ECG signals
    norm_original_data = extracted_data
    for i in range(len(extracted_data)):
        norm_original_data[i][2] = np.array([(float(j) - min(extracted_data[i][2])) /
                                             (max(extracted_data[i][2]) - min(extracted_data[i][2]))
                                             for j in extracted_data[i][2]])
    print('Start slicing data')
    sliced_data = []
    for i, d in enumerate(norm_original_data):
        start = np.where(norm_original_data[i][0] == interval[i][0])[0][0]
        ending = np.where(norm_original_data[i][0] == interval[i][1])[0][0]
        sliced_data.append(align_ecg.sliceup(d, start, ending, patient_id[i], avc_times, p_wave_times, select_point))

    # set as a reference ECG a clear signal. All the ECGs will be aligned with respect to the
    # reference signal
    try:
        patient_ref = patient_id.index(reference_id)
    except ValueError:
        print(f"Participant with ID {reference_id} does not exist. "
              f"Patient with ID: {patient_id[0]} is used as reference")
        patient_ref = patient_id.index(patient_id[0])

    print('Start Alignment')
    # interpolate the ecg signals so that the markers are aligned
    for i in range(len(patient_id)):
        align_ecg.stretch(sliced_data[i], ref=sliced_data[patient_ref])

    # from the time, strain and ecg take only the values that are between the
    # Left Time Marker and Right Time Marker in the txt file
    # this will produce arrays with variable length
    deformation_curve = []
    ecg = []
    time = []
    for i in range(len(norm_original_data)):
        temp1 = []
        temp2 = []
        temp3 = []
        for t, s, e in sliced_data[i].rescaled_slices:
            temp1 = temp1 + s.tolist()
            temp2 = temp2 + e.tolist()
            temp3 = temp3 + t.tolist()
        deformation_curve.append(np.array(temp1))
        ecg.append(np.array(temp2))
        time.append(np.array(temp3))

    # find the patient that have the most samples
    patient_index = np.argmax([len(deformation_curve[i]) for i in range(len(deformation_curve))])

    # interpolate all the other signals so that all the examples have the same number of samples
    interp_ecg, interp_deformation_curve, index, norm_time, interp_time = us.data_interpolation(ecg, deformation_curve,
                                                                                                time, patient_index)
    return interp_ecg, interp_deformation_curve, index, norm_time, interp_time


def plot_gradients(fit_data, interp_time, clustering_results, ids, patient_id, path):
    cluster_labels = {0: 1, 1: 2, 2: 4, 3: 3, 4: 5}
    cluster_colours = {0: "green", 1: "blue", 2: "blueviolet", 3: "darkorange", 4: "red"}

    font = {'family': 'arial',
            'weight': 'normal',
            'size': 20}
    plt.rc('font', **font)
    fig1, ax1 = plt.subplots(figsize=[8, 8])
    fig2, ax2 = plt.subplots(figsize=[8, 8])
    plot_data = []
    for c in range(len(np.unique(clustering_results))):
        centroid = np.zeros(fit_data.shape[1] - 1, dtype=object)
        for counter, id_value in enumerate(ids[c]):
            indice = patient_id.index(id_value)
            if counter == len(ids[c]) - 1:
                trace = go.Scatter(x=interp_time[1:len(interp_time)],
                                   y=np.diff(fit_data[indice]), mode='lines',
                                   opacity=0.5, marker=dict(color=cluster_colours[c]), text='ID: ' + str(id_value),
                                   name="Cluster " + str(cluster_labels[c]), showlegend=True)
            else:
                trace = go.Scatter(x=interp_time[1:len(interp_time)],
                                   y=np.diff(fit_data[indice]), mode='lines',
                                   opacity=0.5, marker=dict(color=cluster_colours[c]), text='ID: ' + str(id_value),
                                   name="Cluster " + str(cluster_labels[c]), showlegend=False)
            plot_data.append(trace)
            ax1.plot(interp_time[1:len(interp_time)], np.diff(fit_data[indice]),
                     cluster_colours[c], alpha=0.6, label="Cluster " + str(cluster_labels[c]))
            centroid += np.diff(fit_data[indice])
        centroid = centroid / len(ids[c])
        ax2.plot(interp_time[1:len(interp_time)], centroid,
                 cluster_colours[c], alpha=0.8, label="Cluster " + str(cluster_labels[c]))

    ax1.set_xlabel("Time (% Cycle)")
    ax1.set_ylabel("LA Strain Rate")
    legend_without_duplicate_labels(ax1)
    fig1.savefig(path + "Gradient.png")
    fig1.savefig(path + "Gradient.svg")
    fig1.show()
    plt.close(fig1)

    ax2.set_xlabel("Time (% Cycle)")
    ax2.set_ylabel("LA Strain Rate")
    ax2.set_title("Centroids of LA Strain Gradients")
    ax2.legend()
    fig2.savefig(path + "Gradient Centroids.png")
    fig2.savefig(path + "Gradient Centroids.svg")
    fig2.show()
    plt.close(fig2)

    layout = go.Layout(title="Gradient Analysis of LA Strain", hovermode='closest',
                       xaxis=dict(title='Time (% of Cycle)'),
                       yaxis=dict(title='LA Strain Rate'), font=dict(size=25))
    fig = dict(data=plot_data, layout=layout)
    plotly.offline.plot(fig, filename=path + 'Gradient.html')


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    legend_attributes = sorted(unique, key=lambda tup: tup[1])
    ax.legend(*zip(*legend_attributes))


def visualize_clustering_results(interp_time, curve, ids, clustering_results, patient_id, centers, path):
    cluster_labels = {0: 1, 1: 2, 2: 4, 3: 3, 4: 5}
    cluster_colours = {0: "green", 1: "blue", 2: "blueviolet", 3: "darkorange", 4: "red"}
    font = {'family': 'arial',
            'weight': 'normal',
            'size': 20}
    plt.rc('font', **font)
    for c in range(len(np.unique(clustering_results))):
        plt.figure(figsize=[8, 8])
        plt.rc('font', **font)
        plot_data = []
        for j in ids[c]:
            indice = patient_id.index(j)
            trace = go.Scatter(x=interp_time, y=curve[indice], mode='lines', opacity=0.5,
                               marker=dict(color='gray'), text='ID: ' + str(j))
            plot_data.append(trace)
            plt.plot(interp_time, curve[indice], "gray", alpha=0.3)

        if centers.all():
            trace = go.Scatter(x=interp_time, y=centers[c], mode="lines", marker=dict(color=cluster_colours[c]),
                               text="Cluster Centroid")
            plot_data.append(trace)
            plt.plot(interp_time, centers[c], cluster_colours[c])

        plt.xlabel("Time (% Cycle)")
        plt.ylabel("Strain (%)")
        plt.title("Cluster " + str(cluster_labels[c]))
        plt.ylim([-20, 80])
        plt.savefig(path + "Groupings of Cluster " + str(cluster_labels[c]) + ".png")
        plt.savefig(path + "Groupings of Cluster " + str(cluster_labels[c]) + ".svg")
        plt.show()
        layout = go.Layout(title='Cluster ' + str(cluster_labels[c]), hovermode='closest',
                           xaxis=dict(title='Time (% of Cycle)', range=[0, 1]),
                           yaxis=dict(title='Strain (%)', range=(-20, 80)),
                           font=dict(size=25))
        fig = dict(data=plot_data, layout=layout)
        plotly.offline.plot(fig, filename=path + 'Groupings of Cluster ' + str(cluster_labels[c]) + '.html')

    fig, ax = plt.subplots(figsize=[8, 8])
    ax.set_title('Cluster Centroids')
    for c in range(len(np.unique(clustering_results))):
        ax.plot(interp_time, centers[c], label='Cluster ' + str(cluster_labels[c]), color=cluster_colours[c],
                linewidth=3)
    ax.set_xlabel('Time (% of Cycle)')
    ax.set_ylabel('Strain (%)')
    # ax.grid()

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

    ax.legend(handles, labels)
    ax.set_ylim(-5, 50)
    plt.savefig(path + 'Cluster Results.png')
    plt.savefig(path + 'Cluster Results.svg')
    plt.show()
    plt.close()


def produce_centroids(predictions, fit_data):
    centers = []
    for i in np.unique(predictions):
        idx = np.where(i == predictions)
        centers.append(np.mean(fit_data[idx], axis=0))
    return centers


def restore_representatives(model, encoded_space, fit_data):
    centroids = model.cluster_centers_
    ind = []
    for i in range(len(centroids)):
        ind.append(np.argmin(np.sum(abs(encoded_space - centroids[i]), axis=1)))
    return fit_data[np.array(ind)]


def reconstruct_curves(model, fit_data, patient_name, time, path):
    if not os.path.exists(path + "Reconstructed Images"):
        os.makedirs(path + "Reconstructed Images")

    predictions = model.predict(fit_data)
    for i in range(len(predictions)):
        plt.figure()
        plt.plot(time, fit_data[i], label='Original Curve')
        plt.plot(time, predictions[i], '--', label='Reconstructed Curve')
        plt.xlabel('Time (% Cycle)')
        plt.ylabel('Strain (%)')
        plt.title('Reconstructed Strain. Patient {}'.format(patient_name[i]))
        plt.legend()
        plt.savefig(path + "Reconstructed Images/Patient " + str(patient_name[i]))
        plt.close()


def plot_pca(clustering_results, fitted_data, pat_id, path):
    pca = PCA(n_components=3)
    pca_components = pca.fit_transform(fitted_data)

    cluster_labels = {0: 1, 1: 2, 2: 4, 3: 3, 4: 5}
    cluster_colours = {0: "green", 1: "blue", 2: "blueviolet", 3: "darkorange", 4: "red"}
    plotted_data = []
    pat_id = np.array(pat_id)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(5):
        # ind = clustering_results.loc[clustering_results['Cluster'] == i + 1].index
        ind = np.where(clustering_results == i)[0]
        trace = go.Scatter3d(x=pca_components[ind, 0], y=pca_components[ind, 1], z=pca_components[ind, 2],
                             mode='markers', opacity=0.45, marker=dict(color=cluster_colours[i]),
                             name='Cluster: ' + str(cluster_labels[i]),
                             hovertext='ID: ' + str(pat_id[ind]) + '\n Cluster: ' + str(cluster_labels[i]))
        plotted_data.append(trace)
        ax.scatter(pca_components[ind, 0], pca_components[ind, 1], pca_components[ind, 2], alpha=0.45,
                   color=cluster_colours[i], label="Cluster " + str(cluster_labels[i]))
    layout = go.Layout(title='PCA of LA Strain (SOM)', hovermode='closest', showlegend=True,
                       scene=go.layout.Scene(xaxis=go.layout.scene.XAxis(title='1st PC'),
                                             yaxis=go.layout.scene.YAxis(title='2nd PC'),
                                             zaxis=go.layout.scene.ZAxis(title='3rd PC')))

    ax.set_xlabel("1st PC")
    ax.set_ylabel("2nd PC")
    ax.set_zlabel("3rd PC")
    ax.set_title("Principal Component Analysis of k-medoids clusters")

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))

    ax.legend(handles, labels)
    plt.savefig(path + "PCA Analysis.png")
    plt.savefig(path + "PCA Analysis.svg")
    plt.show()

    fig = dict(data=plotted_data, layout=layout)
    plotly.offline.plot(fig, filename=path + "/PCA Analysis.html")


def plot_slopes(fit_data, clustering_results, ids, patient_id, systolic_slope, diastolic_slope, path):
    cluster_labels = {0: 1, 1: 2, 2: 4, 3: 3, 4: 5}
    cluster_colours = {0: "green", 1: "blue", 2: "blueviolet", 3: "darkorange", 4: "red"}
    font = {'family': 'arial',
            'weight': 'normal',
            'size': 18}
    plt.rc('font', **font)
    fig_sys, ax_sys = plt.subplots(figsize=[8, 8])
    fig_dia, ax_dia = plt.subplots(figsize=[8, 8])
    plot_data_sys = []
    plot_data_dia = []
    for c in range(len(np.unique(clustering_results))):
        x_dia = []
        y_dia = []
        x_sys = []
        y_sys = []
        for counter, id_value in enumerate(ids[c]):
            indice = patient_id.index(id_value)
            if counter == len(ids[c]) - 1:
                trace = go.Scatter(x=np.array(max(fit_data[indice])),
                                   y=np.abs(np.array(diastolic_slope[indice])), mode='markers',
                                   opacity=0.8, marker=dict(color=cluster_colours[c]), text='ID: ' + str(id_value),
                                   name="Cluster " + str(cluster_labels[c]), showlegend=True)
                plot_data_dia.append(trace)
                trace = go.Scatter(x=np.array(max(fit_data[indice])),
                                   y=np.abs(np.array(systolic_slope[indice])), mode='markers',
                                   opacity=0.8, marker=dict(color=cluster_colours[c]), text='ID: ' + str(id_value),
                                   name="Cluster " + str(cluster_labels[c]), showlegend=True)
                plot_data_sys.append(trace)
            else:
                trace = go.Scatter(x=np.array(max(fit_data[indice])),
                                   y=np.abs(np.array(diastolic_slope[indice])), mode='markers',
                                   opacity=0.8, marker=dict(color=cluster_colours[c]), text='ID: ' + str(id_value),
                                   name="Cluster " + str(cluster_labels[c]), showlegend=False)
                plot_data_dia.append(trace)
                trace = go.Scatter(x=np.array(max(fit_data[indice])),
                                   y=np.abs(np.array(systolic_slope[indice])), mode='markers',
                                   opacity=0.8, marker=dict(color=cluster_colours[c]), text='ID: ' + str(id_value),
                                   name="Cluster " + str(cluster_labels[c]), showlegend=False)
                plot_data_sys.append(trace)

            ax_dia.scatter(np.array(max(fit_data[indice])), np.abs(np.array(diastolic_slope[indice])),
                           c=cluster_colours[c], label="Cluster " + str(cluster_labels[c]))
            ax_sys.scatter(np.array(max(fit_data[indice])), np.abs(np.array(systolic_slope[indice])),
                           c=cluster_colours[c], label="Cluster " + str(cluster_labels[c]))
            x_dia.append(np.array(max(fit_data[indice])))
            y_dia.append(np.abs(np.array(diastolic_slope[indice])))

            x_sys.append(np.array(max(fit_data[indice])))
            y_sys.append(np.abs(np.array(systolic_slope[indice])))
        # confidence_ellipse(np.array(x_dia).astype(float), np.array(y_dia).astype(float).squeeze(), ax_dia,
        #                    facecolor=cluster_colours[c], edgecolor=cluster_colours[c])
        # confidence_ellipse(np.array(x_sys).astype(float), np.array(y_sys).astype(float).squeeze(), ax_sys,
        #                    facecolor=cluster_colours[c],edgecolor=cluster_colours[c])
    ax_sys.set_xlabel("Peak Reservoir Strain (%)")
    ax_sys.set_ylabel("Slope of LA Strain")
    ax_sys.set_title("Slope Analysis during Systole")
    legend_without_duplicate_labels(ax_sys)

    ax_dia.set_xlabel("Peak Reservoir Strain (%)")
    ax_dia.set_ylabel("Slope of LA Strain")
    ax_dia.set_title("Slope Analysis during Diastole")
    legend_without_duplicate_labels(ax_dia)

    fig_sys.savefig(path + "Slope during Systole.png")
    fig_sys.savefig(path + "Slope during Systole.svg")

    fig_dia.savefig(path + "Slope during Diastole.png")
    fig_dia.savefig(path + "Slope during Diastole.svg")

    layout_sys = go.Layout(title="Slope Comparison of LA Strain During Systole", hovermode='closest',
                           xaxis=dict(title='Peak Reservoir Strain (%)'),
                           yaxis=dict(title='Slope of LA Strain'), font=dict(size=25))
    fig1 = dict(data=plot_data_sys, layout=layout_sys)
    plotly.offline.plot(fig1, filename=path + 'Slope during Systole.html')

    layout_dia = go.Layout(title="Slope Comparison of LA Strain During Diastole", hovermode='closest',
                           xaxis=dict(title='Peak Reservoir Strain (%)'),
                           yaxis=dict(title='Slope of LA Strain'), font=dict(size=25))
    fig2 = dict(data=plot_data_dia, layout=layout_dia)
    plotly.offline.plot(fig2, filename=path + 'Slope during Diastole.html')


def confidence_ellipse(x, y, ax, facecolor=None, edgecolor=None, a=0.5):

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    ell_radius_x = (max(x) - min(x))/2
    ell_radius_y = (max(y) - min(y))/2
    center = ((max(x) + min(x))/2, (max(y) + min(y))/2)
    x_max_index = np.argmax(x)
    x_min_index = np.argmin(x)

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y.reshape(-1, 1))

    # rotation = float(np.degrees(np.arctan((y[x_max_index]-y[x_min_index])/(x[x_max_index]-x[x_min_index]))))
    rotation = model.coef_
    print(rotation)
    ellipse = Ellipse(center, width=ell_radius_x*2, height=ell_radius_y*2, angle=-rotation,
                      facecolor=facecolor, edgecolor=edgecolor, alpha=a)

    return ax.add_patch(ellipse)
