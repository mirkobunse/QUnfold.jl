import astropy.units as u
import click
import h5py
import numpy as np
import pandas as pd
from fact.io import read_h5py, read_simulated_spectrum
from fact_funfolding.binning import logspace_binning
from irf.collection_area import collection_area

@click.command()
@click.argument('gamma_file')
@click.argument('corsika_file')
@click.argument('output_file')
@click.option('--threshold', type=float, default=0.8, help='decision threshold for gamma predictions')
@click.option('--theta2_cut', type=float, default=0.025, help='theta2 cut for on/off predictions')
@click.option('--e_min', type=float, default=10**2.4, help='minimum energy (default: 10^2.4)')
@click.option('--e_max', type=float, default=10**4.2, help='minimum energy (default: 10^4.2)')
@click.option('--n_bins', type=int, default=12, help='number of bins (default: 12)')
def main(gamma_file, corsika_file, output_file, threshold, theta2_cut, e_min, e_max, n_bins):
    '''Foo bar'''
    print(f'Reading simulated gammas from {gamma_file}')
    query = 'gamma_prediction > {} and theta_deg**2 < {}'.format(threshold, theta2_cut)
    gammas = read_h5py(gamma_file, key='events').query(query)
    with h5py.File(gamma_file, 'r') as f:
        sample_fraction = f.attrs.get('sample_fraction', 1.0)
        print('Using sampling fraction of {:.3f}'.format(sample_fraction))

    print(f'Reading CORSIKA headers from {corsika_file}')
    simulated_spectrum = read_simulated_spectrum(corsika_file)
    corsika_events = read_h5py(
        corsika_file,
        key='corsika_events',
        columns=['total_energy'],
    )

    bins_true = logspace_binning(e_min * u.GeV, e_max * u.GeV, 1 * u.GeV, n_bins)
    a_eff, bin_center, bin_width, a_low, a_high = collection_area(
        corsika_events.total_energy.values,
        gammas['corsika_event_header_total_energy'].values,
        impact=simulated_spectrum['x_scatter'],
        bins=bins_true.to_value(u.GeV),
        sample_fraction=sample_fraction,
    )
    df = pd.DataFrame({
        'bin': np.arange(n_bins+2), # index 0 and n_bins+1 hold everything outside
        'a_eff': a_eff,
        'bin_center': bin_center,
        'bin_width': bin_width,
        'conf_lower': a_low,
        'conf_upper': a_high,
        'e_min': bin_center - bin_width/2,
        'e_max': bin_center + bin_width/2
    })

    print(f'Writing effective areas to {output_file}')
    df.to_csv(output_file)

if __name__ == '__main__':
    main()
