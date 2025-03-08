import numpy as np


def driver(signal_df, n_components=3, signal_column='Open'):
    # iterate over unique days in the df
    dates = signal_df.Date.unique()
    label_arr = np.zeros((signal_df.shape[0], 2))
    label_arr[:, 0] = signal_df.EpochTime.values
    for d_idx in range(len(dates)):
        # filter on this date
        this_date = dates[d_idx]
        insert_idx = (signal_df['Date'] == this_date)
        signal = signal_df[signal_column].loc[insert_idx].values

        # compute search space for signal
        search_space = compute_search_space(signal)

        # compute growth/decay function and coefficients
        exp_func, _ = compute_growth_coeff(signal, search_space)

        # remove behavior from signal
        no_growth_signal = signal - exp_func

        # compute FFT on growth-less signal
        fft_signal, _ = compute_fft(no_growth_signal, n_components)

        # combine for final signal
        final_signal = exp_func + fft_signal

        # compute differentials of the clean signal to obtain our labels
        clean_signal_diff = np.gradient(final_signal)

        # all points with a positive gradient value are labelled 1
        label_slice = np.zeros((signal.shape[0], 2))
        label_slice[:, 0] = signal_df.EpochTime.values[insert_idx]
        label_slice[clean_signal_diff > 0, 1] = 1
        label_arr[insert_idx] = label_slice

    label_df = pd.DataFrame(label_arr, columns=['EpochTime', 'Label'])
    return label_df


def compute_search_space(signal):
    if signal[-1] > signal[0]:
        # if signal is increasing
        A_values = np.arange(.1, 10.1, .1)
    elif signal[-1] > signal[0]:
        # if signal is decreasing
        A_values = np.arange(-10, 0, .1)
    else:
        # if signal is flat
        A_values = np.arange(-1, 1.1, .1)

    B_values = np.arange(.1, 3.1, .1)

    return np.array(np.meshgrid(A_values, B_values)).T.reshape(-1, 2)


def compute_growth_coeff(signal, search_space):
    # Step 1: Convert the Search Space into a Function Space  ###
    t = np.arange(0, signal.shape[0]).reshape(-1, 1)
    # stack time column vectors into a matrix that's len(time values) x len(combos)
    func_space = np.repeat(t, search_space.shape[0], axis=1)
    # raise time values to B powers in search space
    func_space = func_space ** search_space[:, 1]
    # scale functions by A coefficients
    func_space = search_space[:, 0] * func_space
    # add C term
    func_space += signal[0]

    # Step 2: Compute Error Space
    signal_mat = np.repeat(signal.reshape(-1, 1), search_space.shape[0], axis=1)
    error_space = np.sum(np.abs(signal_mat - func_space), axis=0)

    # Step 3: Find Combo in Search Space w/ Lowest Error
    best_idx = np.argmin(error_space)
    best_combo = search_space[best_idx]
    best_func = func_space[:, best_idx]

    return best_func, [best_combo[0], best_combo[1], signal[0]]


def compute_fft(signal, n_components=8):
    # compute FFT on signal
    n = signal.shape[0]  # get length of signal
    fhat = np.fft.fft(signal, n)  # compute FFT on noisy data
    PSD = fhat * np.conj(fhat) / n  # power spectral density -- multiply complex term by its conjugate and divide by n

    # compute values from FFT
    freq = (1 / n) * np.arange(n)  # frequency from full FFT
    amp = 2 / n * np.abs(fhat)  # amplitude from full FFT
    phase_shift = np.angle(fhat)  # phase shift from full FFT

    # down-sample the computed values from the full FFT
    L = np.arange(1, np.floor(n / 2), dtype='int')
    PSD_reduced = PSD[L]
    freq_reduced = freq[L]
    amp_reduced = amp[L]
    phase_shift_reduced = phase_shift[L]

    # get Fourier Series values
    min_PSD = np.sort(PSD_reduced)[-n_components]
    keep_idx_reduced = PSD_reduced > min_PSD
    freq_clean = freq_reduced[keep_idx_reduced]
    amp_clean = amp_reduced[keep_idx_reduced]
    phase_shift_clean = phase_shift_reduced[keep_idx_reduced]

    # compute inverse FFT for Clean Signal
    keep_idx_full = PSD > min_PSD
    fhat_clean = fhat * keep_idx_full
    fft_clean_signal = np.real(np.fft.ifft(fhat_clean))

    return fft_clean_signal, [amp_clean, freq_clean, phase_shift_clean]


if __name__ == '__main__':
    import pandas as pd
    import plotting.plot_shortcuts as ps
    import arima

    df = pd.read_csv('..\\data\\qqq_2022.csv')
    # df = pd.read_csv('../data/qqq_2022.csv')
    sig_col = 'Open'
    raw_signal_df = df[['EpochTime', 'Date', sig_col]]  # .loc[df.Date.values == df.Date.unique()[1]]

    # component analysis test
    c_num = 20
    label_df = driver(signal_df=raw_signal_df, n_components=c_num, signal_column=sig_col)
    plt = ps.plot_label_over_signal(raw_signal_df, label_df=label_df)
    plt.show()

    # Train ARIMA and get the forecast values
    forecast_df = arima.train_test(raw_signal_df, sig_col)

    # From the forecasts, classify each point as increasing (1) or decreasing (0)
    diffs = np.diff(forecast_df.values, axis=0)
    yhat = np.where(diffs > 0, 1, 0)
    
    print('Forecast Values:')
    print(forecast_df[:10])
    print('\nForecast Classifications:')
    print(yhat[:10])
