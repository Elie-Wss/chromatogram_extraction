# visualization_functions.py
import matplotlib.pyplot as plt
import numpy as np
import plotly 
import plotly.graph_objects as go

def visualizer(chromato_obj, mod_time = 1.25, rt1 = None, rt2 = None, rt1_window = 5, rt2_window = 0.1, plotly = False, title = "", points = None, radius=None, pt_shape = ".", log_chromato=True, casnos_dict=None, contour=[], center_pt=None, center_pt_window_1 = None, center_pt_window_2 = None, save=False, show=True):
    r"""Plot mass spectrum

    Parameters
    ----------
    chromato_obj :
        chromato_obj=(chromato, time_rn)
    rt1: optional
        Center the plot in the first dimension around rt1
    rt2: optional
        Center the plot in the second dimension around rt2
    rt1_window: optional
        If rt1, window size in the first dimension around rt1
    rt2_window: optional
        If rt2, window size in the second dimension around rt2
    points: optional
        Coordinates to displayed on the chromatogram
    radius: optional
        If points, dislay their radius (blobs detection)
    pt_shape: optional
        Shape of the points to be displayed.
    log_chromato: optional
        Apply logarithm function to the chromato before visualization
    contour: optional
        Displays stitch outlines
    center_pt: optional
        Center the plot around center_pt coordinates
    center_pt_window_1: optional
        If center_pt, window size in the first dimension around center_pt first coordinate
    center_pt_window_2: optional
        If center_pt, window size in the second dimension around center_pt second coordinate
    casnos_dict: optional
        If center_pt, window size in the second dimension around center_pt second coordinate
    title: optional
        Title of the plot
    Returns
    -------
    Examples
    --------
    >>> import plot
    >>> import read chroma
    >>> import utils
    >>> chromato_obj = read_chroma.read_chroma(filename, mod_time)
    >>> chromato,time_rn,spectra_obj = chromato_obj
    >>> matches = matching.matching_nist_lib(chromato_obj, spectra, some_pixel_coordinates)
    >>> casnos_dict = utils.get_name_dict(matches)
    >>> coordinates_in_time = projection.matrix_to_chromato(u,time_rn, mod_time,chromato.shape)
    >>> plot.visualizer(chromato_obj=(chromato, time_rn), mod_time=mod_time, points=coordinates_in_time, casnos_dict=casnos_dict)
    """
    chromato, time_rn = chromato_obj
    shape = chromato.shape
    X = np.linspace(time_rn[0], time_rn[1], shape[0])
    Y = np.linspace(0, mod_time, shape[1])
    if (rt1 is not None and rt2 is not None):
        rt1minusrt1window = rt1 - rt1_window
        rt1plusrt1window = rt1 + rt1_window
        rt2minusrt2window = rt2 - rt2_window
        rt2plusrt2window = rt2 + rt2_window
        if (rt1minusrt1window < time_rn[0]):
            rt1minusrt1window = time_rn[0]
            rt1plusrt1window = rt1 + rt1_window
        if (rt1plusrt1window > time_rn[1]):
            rt1plusrt1window = time_rn[1]
            rt1minusrt1window = rt1 - rt1_window
        if (rt2minusrt2window < 0):
            rt2minusrt2window = 0
            rt2plusrt2window = rt2 + rt2_window
        if (rt2plusrt2window > mod_time):
            rt2plusrt2window = mod_time
            rt2minusrt2window = rt2 - rt2_window
        position_in_chromato = np.array([[rt1minusrt1window, rt2minusrt2window], [rt1plusrt1window, rt2plusrt2window]])
        indexes = chromato_to_matrix(position_in_chromato,time_rn=time_rn, mod_time=mod_time, chromato_dim=shape)
        indexes_in_chromato = matrix_to_chromato(indexes,time_rn=time_rn, mod_time=mod_time, chromato_dim=shape)
        chromato = chromato[indexes[0][0]:indexes[1][0], indexes[0][1]:indexes[1][1]]
        X = np.linspace(rt1minusrt1window, rt1plusrt1window, indexes[1][0] - indexes[0][0])
        Y = np.linspace(rt2minusrt2window, rt2plusrt2window, indexes[1][1] - indexes[0][1])
    elif (center_pt_window_1 and center_pt_window_2):
        center_pt1_minusrt1window = center_pt[0] - center_pt_window_1
        center_pt1_plusrt1window =  center_pt[0] + center_pt_window_1
        center_pt2_minusrt2window =  center_pt[1] - center_pt_window_2
        center_pt2_plusrt2window =  center_pt[1] + center_pt_window_2
        if (center_pt1_minusrt1window < 0):
            center_pt1_minusrt1window = 0
            center_pt1_plusrt1window = 2 * center_pt[0]
        if (center_pt1_plusrt1window >= shape[0]):
            center_pt1_plusrt1window = shape[0] - 1
            center_pt1_minusrt1window = center_pt[0] - abs(center_pt[0] - center_pt1_plusrt1window)
        if (center_pt2_minusrt2window < 0):
            center_pt2_minusrt2window = 0
            center_pt2_plusrt2window = 2 * center_pt[1]
        if (center_pt2_plusrt2window >= shape[1]):
            center_pt2_plusrt2window = shape[1] - 1
            center_pt2_minusrt2window = center_pt[1] - abs(center_pt[1] - center_pt2_plusrt2window)

        chromato = chromato[center_pt1_minusrt1window:center_pt1_plusrt1window + 1, center_pt2_minusrt2window:center_pt2_plusrt2window + 1]
        position_in_chromato = np.array([[center_pt1_minusrt1window, center_pt2_minusrt2window], [center_pt1_plusrt1window, center_pt2_plusrt2window]])
        indexes = matrix_to_chromato(position_in_chromato,time_rn=time_rn, mod_time=mod_time, chromato_dim=shape)
        #indexes_in_chromato = matrix_to_chromato(indexes,time_rn=time_rn, mod_time=mod_time, chromato_dim=shape)
        indexes_in_chromato=indexes

        X = np.linspace(indexes[0][0], indexes[1][0], chromato.shape[0])
        Y = np.linspace(indexes[0][1], indexes[1][1], chromato.shape[1])

        indexes = np.array([[center_pt1_minusrt1window, center_pt2_minusrt2window], [center_pt1_plusrt1window + 1, center_pt2_plusrt2window + 1]])
    if (log_chromato):
        chromato = np.log(chromato)
    chromato = np.transpose(chromato)
    fig, ax = plt.subplots()

    #tmp = ax.pcolormesh(X, Y, chromato)
    tmp = ax.contourf(X, Y, chromato)
    plt.colorbar(tmp)
    if (title != ""):
        plt.title(title)
    if (points is not None):
        if ((rt1 and rt2) or (center_pt_window_1 and center_pt_window_2)):
            tmp = []
            point_indexes = chromato_to_matrix(points,time_rn=time_rn, mod_time=mod_time, chromato_dim=shape)
            for i, point in enumerate(point_indexes):
                if (point_is_visible(point, indexes)):
                    tmp.append(points[i])

            points = np.array(tmp)
        if (radius is not None and len(points) > 0):
            for i in range(len(points)):
                c = plt.Circle((points[i][0], points[i][1]), radius[i] / shape[1] , color="red", linewidth=2, fill=False)
                ax.add_patch(c)
        if (len(points) > 0):
            if (casnos_dict != None):
                mol_name = []
                scatter_list = []
                comp_list = list(casnos_dict.keys())
                nb_comp = len(comp_list)
                cmap = get_cmap(nb_comp)
                for i, casno in enumerate(comp_list):
                    tmp_pt_list = []
                    for pt in casnos_dict[casno]:
                        if (not((rt1 and rt2) or (center_pt_window_1 and center_pt_window_2)) or point_is_visible(pt, indexes_in_chromato)):
                            print(casno)
                            tmp_pt_list.append(pt)
                    '''x_pts = np.array(casnos_dict[casno])[:,0]
                    y_pts = np.array(casnos_dict[casno])[:,1]'''
                    if len(tmp_pt_list) == 0:
                        continue
                    print("----")

                    mol_name.append(comp_list[i])
                    tmp_pt_list = np.array(tmp_pt_list)
                    x_pts = tmp_pt_list[:,0]
                    y_pts = tmp_pt_list[:,1]
                    tmp = ax.scatter(x_pts,y_pts, c=cmap(i), marker=pt_shape, cmap='hsv')
                    scatter_list.append(tmp)
                print(mol_name)
                plt.legend(scatter_list,
                    mol_name,
                    scatterpoints=1, fontsize=8, ncol=1, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand")
            else:
                ax.plot(points[:,0], points[:,1], "r" + pt_shape)
    if (len(contour)):
        if (center_pt_window_1 and center_pt_window_2):
            indexes_in_chromato = matrix_to_chromato(indexes,time_rn=time_rn, mod_time=mod_time, chromato_dim=shape)
            tmp = []
            for i in range(len(contour)):
                if (point_is_visible(contour[i], indexes_in_chromato)):
                    tmp.append(contour[i])
            tmp=np.array(tmp)
            ax.plot(tmp[:,0], tmp[:,1], "b.")
        else:
            ax.plot(contour[:,0], contour[:,1], "b.")
    if (save):
        plt.savefig("figs/chromato_" + title + ".png")

    if show:
        plt.show()
        if (plotly):
            fig = go.Figure(data =
            go.Contour(
                z=np.transpose(chromato),
                x = np.linspace(time_rn[0], time_rn[1], shape[0]),
                y = np.linspace(0, mod_time, shape[1])
            ))
            fig.show()

def chromato_to_matrix(points, time_rn, mod_time, chromato_dim):
    # Converts chromatogram data to a matrix (placeholder implementation)
    if (points is None):
        return None
    #return np.rint(np.column_stack(((points[:,0] -  time_rn[0]) * chromato_dim[0] / (time_rn[1] - time_rn[0]), points[:,1] / mod_time * chromato_dim[1]))).astype(int)
    return np.rint(np.column_stack(((points[:,0] -  time_rn[0]) * (chromato_dim[0] - 1) / (time_rn[1] - time_rn[0]), points[:,1] / mod_time * (chromato_dim[1] - 1)))).astype(int)

def matrix_to_chromato(points, time_rn, mod_time, chromato_dim):
    # Converts matrix data back to chromatogram format (placeholder implementation)
    if (points is None):
        return None
    #return np.column_stack((points[:,0] * (time_rn[1] - time_rn[0]) / (chromato_dim[0]) + time_rn[0], points[:,1] * mod_time / chromato_dim[1]))
    return np.column_stack((points[:,0] * (time_rn[1] - time_rn[0]) / (chromato_dim[0] - 1) + time_rn[0], points[:,1] * mod_time / (chromato_dim[1] - 1)))

def point_is_visible(point, indexes):
    # Determine if the point should be visible (placeholder)
    x,y = point[0], point[1]
    if (x <= indexes[0][0] or x >= indexes[1][0] or y <= indexes[0][1] or y >= indexes[1][1]):
        return False
    return True

def get_cmap(n, name='hsv'):
    """
    Return a function that maps each index in 0, 1, ..., n-1 to a distinct RGB color.
    """
    return plt.get_cmap(name, n)

# Add any additional visualization helper functions if needed.