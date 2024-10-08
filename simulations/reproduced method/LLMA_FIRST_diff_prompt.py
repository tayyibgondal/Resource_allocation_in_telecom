from llama_cpp import Llama

import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim

#import seaborn as sns


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

np.set_printoptions(precision=3)


def loc_init(Size_area, Dist_TX_RX, Num_D2D, Num_Ch):
    # Generate random locations for D2D transmitters within the area
    # Size_area: Defines the size of the square area where the transmitters are located
    # Dist_TX_RX: Maximum distance allowed between transmitter and receiver
    # Num_D2D: Number of device-to-device (D2D) (users)
    # Num_Ch: Number of channels  #TODO
    
    # Randomly generate D2D transmitter locations. np.random.rand generates values in [0, 1), so we shift to [-0.5, 0.5)
    # Then, scale by Size_area to ensure the locations are within the defined area.
    tx_loc = Size_area * (np.random.rand(Num_D2D, 2) - 0.5)  #TODO
    
    # Initialize an array to store D2D receiver locations
    # We have Num_D2D + 1 receivers because the last one is for the Cellular User Equipment (CUE)
    rx_loc = np.zeros((Num_D2D + 1, 2))  #TODO
    
    # For each D2D transmitter, generate a feasible location for its corresponding receiver
    for i in range(Num_D2D):
        # Call Feasible_Loc_Init to get a random location for the receiver based on the current transmitter's location
        # This function ensures the receiver is within a certain distance (Dist_TX_RX) from the transmitter and within bounds.
        temp_chan = Feasible_Loc_Init(tx_loc[i, :], Size_area, Dist_TX_RX)
        # Assign the generated receiver location to rx_loc
        rx_loc[i, :] = temp_chan

    # Generate random locations for CUE transmitters
    # CUE (Cellular User Equipment) locations are generated similarly to D2D transmitters, scaled within the Size_area.
    tx_loc_CUE = Size_area * (np.random.rand(Num_Ch, 2) - 0.5)  #TODO

    # Return the receiver locations (including the extra one for CUE), the D2D transmitter locations, and the CUE transmitter locations
    return rx_loc, tx_loc, tx_loc_CUE



## Check the feasibility of generated location for the receiver
def Feasible_Loc_Init(Cur_loc, Size_area, Dist_TX_RX):
    # Generate a random distance vector from the current location (Cur_loc).
    # The vector is scaled within [-Dist_TX_RX, Dist_TX_RX] for both x and y coordinates.
    # (np.random.rand(1, 2) - 0.5) generates values in the range [-0.5, 0.5].
    # Multiplying by 2 * Dist_TX_RX scales this to the desired range.
    temp_dist = 2 * Dist_TX_RX * (np.random.rand(1, 2) - 0.5)
    
    # Calculate the temporary channel (location) by adding the random distance vector to the current location.
    temp_chan = Cur_loc + temp_dist

    # Check if the generated location is valid:
    # The while loop ensures that the new location:
    # 1. Stays within the boundaries of the area (|x|, |y| ≤ Size_area / 2).
    # 2. Maintains the distance constraint (less than or equal to Dist_TX_RX).
    while (np.max(abs(temp_chan)) > Size_area / 2) | (np.linalg.norm(temp_dist) > Dist_TX_RX):
        # If the conditions are not met, generate a new random distance vector.
        temp_dist = 2 * Dist_TX_RX * (np.random.rand(1, 2) - 0.5)
        # Recalculate the temporary channel location.
        temp_chan = Cur_loc + temp_dist
    
    # Return the feasible location that meets all constraints.
    return temp_chan



## Generate sample data for the channel
def ch_gen(Size_area, D2D_dist, Num_D2D, Num_Ch, Num_samples, PL_alpha=38., PL_const=34.5):
    '''
    The function ch_gen generates synthetic channel data for D2D and CUE communication links
    by simulating the positions of transmitters and receivers within a defined area. It 
    calculates channel gains incorporating path loss and multi-path fading effects.
    '''
    # Initialize empty lists to store channel data and user locations
    ch_w_fading = []  # Stores the channel gains with fading effects
    rx_loc_mat = []   # Stores receiver locations for each sample
    tx_loc_mat = []   # Stores transmitter locations for each sample
    CUE_loc_mat = []  # Stores CUE transmitter locations for each sample

    # Perform initial location setup for transmitters and receivers
    # This initialization is done once and used as a base to generate channels by adjusting locations
    rx_loc, tx_loc, tx_loc_CUE = loc_init(Size_area, D2D_dist, Num_D2D, Num_Ch)

    # Loop through the number of samples to generate channel data
    for i in range(Num_samples):
        # Reinitialize locations of transmitters and receivers for each sample
        rx_loc, tx_loc, tx_loc_CUE = loc_init(Size_area, D2D_dist, Num_D2D, Num_Ch)

        # Temporary list to store channel gains for each channel (frequency band)
        ch_w_temp_band = []  #(numchan, numtrans, numreceiver) #TODO
        for j in range(Num_Ch):
            # Combine D2D transmitters and the current CUE transmitter for channel calculation
            tx_loc_with_CUE = np.vstack((tx_loc, tx_loc_CUE[j]))  # (1+num_d2d, 2)

            # Calculate the distance vector between each receiver and each transmitter
            # rx_loc is reshaped to (Num_D2D + 1, 1, 2) for proper broadcasting during subtraction
            dist_vec = rx_loc.reshape(Num_D2D + 1, 1, 2) - tx_loc_with_CUE  # (Num_D2D + 1, Num_D2D + 1)
            # Calculate Euclidean distance between each pair (receiver-transmitter)
            dist_vec = np.linalg.norm(dist_vec, axis=2)
            # Prevent distances from being too small (minimum distance threshold of 3 meters) #TODO
            dist_vec = np.maximum(dist_vec, 3)

            # Calculate path loss (in dB) using the log-distance path loss model (without shadowing)
            # PL_alpha is the path loss exponent, and PL_const is the path loss constant
            pu_ch_gain_db = - PL_const - PL_alpha * np.log10(dist_vec)
            # Convert path loss from dB to linear scale to obtain the channel gain
            pu_ch_gain = 10 ** (pu_ch_gain_db / 10)

            # Generate multi-path fading using Rayleigh distribution (sum of squared Gaussian variables)
            # Generates random fading coefficients for each link between transmitters and receivers
            multi_fading = (
                0.5 * np.random.randn(Num_D2D + 1, Num_D2D + 1) ** 2 +
                0.5 * np.random.randn(Num_D2D + 1, Num_D2D + 1) ** 2
            )

            # Calculate the final channel gain by multiplying path loss gain with fading coefficients
            # Use np.maximum to ensure the channel gain does not fall below a very small threshold (avoid numerical errors)
            final_ch = np.maximum(pu_ch_gain * multi_fading, np.exp(-30)) # (Num_D2D + 1, Num_D2D + 1)

            # Store the transposed channel matrix for the current channel (frequency band)
            ch_w_temp_band.append(np.transpose(final_ch))

        # Append the channel gain matrix for all channels of the current sample
        ch_w_fading.append(ch_w_temp_band)  #(samples, num_chan, Num_D2D + 1, Num_D2D + 1)
        # Store the locations of receivers, transmitters, and CUE transmitters for the current sample
        rx_loc_mat.append(rx_loc)   #(samples, Num_D2D + 1, Num_D2D + 1)
        tx_loc_mat.append(tx_loc)   #(samples, Num_D2D + 1, Num_D2D + 1)
        CUE_loc_mat.append(tx_loc_CUE)    #(samples, Num_D2D + 1, Num_D2D + 1)

    # Convert the collected channel gains and locations into numpy arrays for further processing
    return np.array(ch_w_fading), np.array(rx_loc_mat), np.array(tx_loc_mat), np.array(CUE_loc_mat)


## Simulation setting for sample channel data

Size_area = 50.0
D2D_dist = 10
Num_user = 2
Num_channel = 1
num_samples_tr = 30


## Calculate Rate and Energy Efficiency for D2D Communication
def cal_rate_NP(channel, tx_power_in, noise, DUE_thr, I_thr, P_c):
    '''
    The function cal_rate_NP calculates the sum rate (SE) and energy efficiency (EE) 
    of Device-to-Device (D2D) communications given a channel matrix, powers and transmission parameters.
    It evaluates whether the communication meets specific thresholds for D2D user equipment (DUE)
    and the cellular user equipment (CUE).
    '''
    
    # Get the number of samples, channels, and D2D users from the channel shape
    num_sample = channel.shape[0]  # Total number of samples (simulations)
    num_channel = channel.shape[1]  # Total number of channels (frequency bands)
    num_D2D_user = channel.shape[2] - 1  # Number of D2D users (excluding CUE)

    ## Initialization of cumulative metrics
    tot_SE = 0  # Total Sum Rate (SE)
    tot_EE = 0  # Total Energy Efficiency (EE)
    DUE_violation = 0  # Count of DUE violations
    CUE_violation = 0  # Count of CUE violations

    ## Append transmission power for each channel
    # Create a transmission power array that includes CUE's transmission power set to zero
    tx_power = np.hstack((tx_power_in, 0 * np.ones((tx_power_in.shape[0], 1, num_channel))))

    # Loop through each sample
    for i in range(num_sample):
        cur_cap = 0  # Current capacity for this sample
        DUE_mask = 1  # Mask to track DUE threshold satisfaction
        CUE_mask = 1  # Mask to track CUE threshold satisfaction

        # Loop through each channel
        for j in range(num_channel):
            cur_ch = channel[i][j]  # Get the channel for current sample and channel
            cur_power = tx_power[i, :, j]  # Get the corresponding transmission power
            cur_power = np.array([cur_power])  # Reshape power for calculation

            # Calculate the capacity for this sample and channel
            cur_ch_cap = cal_RATE_one_sample_one_channel(cur_ch, cur_power, noise)
            # Calculate the interference for CUE from this channel
            inter = cal_CUE_INTER_one_sample_one_channel(cur_ch, cur_power)

            cur_cap = cur_cap + cur_ch_cap[0]  # Accumulate the capacity for D2D users
            CUE_mask = CUE_mask * (inter[0, num_D2D_user] <= I_thr)  # Check CUE interference threshold

        # Check D2D user capacity against thresholds
        for j in range(num_D2D_user):
            DUE_mask = DUE_mask * (cur_cap[j] >= DUE_thr)  # Update mask for DUE threshold satisfaction

        # Calculate the sum of rates and energy efficiency for D2D users
        D2D_SE_sum = np.sum(cur_cap[:-1]) * CUE_mask * DUE_mask  # Sum Rate
        D2D_EE_sum = np.sum(cur_cap[:-1] / (np.sum(tx_power_in[i], axis=1) + P_c)) * CUE_mask * DUE_mask  # Energy Efficiency

        # Count violations for CUE and DUE
        if CUE_mask == 0:
            CUE_violation = CUE_violation + 1  # Increment CUE violation count

        if DUE_mask == 0:
            DUE_violation = DUE_violation + 1  # Increment DUE violation count

        # Accumulate total metrics
        tot_SE = tot_SE + D2D_SE_sum  # Update total SE
        tot_EE = tot_EE + D2D_EE_sum  # Update total EE

    # Calculate averages for SE and EE
    tot_SE = tot_SE / num_D2D_user / num_sample  # Average SE
    tot_EE = tot_EE / num_D2D_user / num_sample  # Average EE

    # Calculate probabilities of violations
    PRO_DUE_vio = DUE_violation / (num_sample)  # Probability of DUE violation
    PRO_CUE_vio = CUE_violation / (num_sample)  # Probability of CUE violation

    return tot_SE, tot_EE, PRO_CUE_vio, PRO_DUE_vio  # Return the results



ch_mat, rx_mat, tx_mat, CUE_mat = ch_gen(Size_area, D2D_dist, Num_user, Num_channel, int(10**4))
tx_power_in = 10**2.0*np.ones((ch_mat.shape[0], 2, 1))



def all_possible_tx_power(num_channel, num_user, granuty):
    """
    Generate all possible transmission power configurations for a given number of channels and users,
    constrained by a specified granularity.

    Parameters:
    - num_channel: The number of channels.
    - num_user: The number of users.
    - granuty: The maximum value for transmission power levels (not including granuty itself).

    Returns:
    - A numpy array containing valid power configurations.
    """

    # Create a list of arrays representing possible power levels for each user and channel.
    # Each user/channel can have values ranging from 0 to granuty-1.
    items = [np.arange(granuty)] * (num_user * num_channel)  #(granularity-1, num_user*numchan)

    # Generate all combinations of power levels for each user and channel using Cartesian product.
    temp_power = list(itertools.product(*items))  # (num_combinations, num_user*numchan)

    # Reshape the list of combinations into a 3D numpy array:
    # Shape will be (number of combinations, num_user, num_channel)
    temp_power = np.reshape(temp_power, (-1, num_user, num_channel))  # (num_combinations, num_user, numchan)

    # Sum the transmission powers across all channels for each configuration.
    power_check = np.sum(temp_power, axis=2)   # (num_combinations, num_user)

    # Create a flag to check if the normalized power for each configuration is less than or equal to 1.
    # This ensures that the total power does not exceed a certain limit.
    flag = (power_check / (granuty - 1) <= 1).astype(int)   # (num_combinations, num_user) 

    # Check that exactly one user has a non-zero power allocation in each configuration.
    flag = (np.sum(flag, axis=1) / num_user == 1).astype(int)   # (num_combinations,)   #TODO

    # Reshape the flag to have shape (number of combinations, 1) for later multiplication.
    flag = np.reshape(flag, (-1, 1))  #(number of combinations, 1)

    # Reshape temp_power to 2D for filtering:
    # Shape will be (number of combinations, num_user * num_channel)
    temp_power_1 = np.reshape(temp_power, (-1, num_user * num_channel))   # (num_combinations, num_user, numchan)

    # Filter valid power configurations by multiplying with the flag.
    # Invalid configurations will be zeroed out.
    temp_power = temp_power_1 * flag

    # Reshape back to original dimensions and normalize the power values.
    power = np.reshape(temp_power, (-1, num_user, num_channel)) / (granuty - 1)   # (num_combinations, num_user, numchan)

    # Initialize a list to collect valid power configurations.
    power_mat = []

    # Iterate over each configuration to filter out those with a sum of zero.
    for i in range(power.shape[0]):
        sum_val = np.sum(power[i])
        # Only include configurations where the sum of powers is not zero.
        if sum_val != 0:
            power_mat.append(power[i])

    # Return the valid power configurations as a numpy array.
    return np.array(power_mat)   # (num_valid_combinations, num_user, numchan)


def optimal_power(channel, tx_max, noise, DUE_thr, I_thr, P_c, tx_power_set, opt="SE"):
    '''
    This function calculates the optimal transmission power for Device-to-Device (D2D) communications 
    under certain constraints and objectives (either maximizing Spectral Efficiency (SE) or Energy Efficiency (EE)).
    It also takes into account interference to Cellular User Equipment (CUE) and ensures that D2D users meet
    minimum data rate (capacity) thresholds.
    
    Parameters:
    - channel: A 3D array representing the channel gains for each user and sample.
    - tx_max: Maximum transmission power allowed for each user.
    - noise: Noise level affecting the transmission.
    - DUE_thr: Minimum capacity threshold for D2D users.
    - I_thr: Interference threshold for CUE (Cellular User Equipment) caused by D2D communications.
    - P_c: Constant power consumed regardless of transmission.
    - tx_power_set: Set of transmission power configurations to consider.
    - opt: Objective for optimization, either "SE" for Spectral Efficiency or "EE" for Energy Efficiency.
    
    Returns:
    - tot_SE: Total Spectral Efficiency.
    - tot_EE: Total Energy Efficiency.
    - PRO_CUE_vio: Probability of CUE violations.
    - PRO_DUE_vio: Probability of D2D violations.
    - chan_infea_mat: Channels that were found infeasible.
    '''
    
    # Extract the number of channels, D2D users, and samples from the channel data
    num_samples = channel.shape[0]
    num_channel = channel.shape[1]
    num_D2D_user = channel.shape[2] - 1  # Last user is the CUE
    
    # Initialize total Spectral Efficiency and lists to store power settings and infeasible channels
    tot_SE = 0
    power_mat_SE = []
    chan_infea_mat = []

    # Iterate through each sample to calculate optimal power
    for i in range(num_samples):
        cur_cap = 0  # Current capacity
        DUE_mask = 1  # Mask to check if DUE threshold is met
        CUE_mask = 1  # Mask to check if CUE interference is acceptable
        
        # Prepare the transmission power for the current configuration
        # Append a zero power for the CUE (last user) since we're focusing on D2D users
        tx_power = tx_max * np.hstack((tx_power_set, 0 * np.ones((tx_power_set.shape[0], 1, num_channel))))

        # Loop through each channel to calculate capacity
        for j in range(num_channel):
            cur_ch = channel[i][j]  # Current channel matrix for this sample and channel
            
            # Calculate the channel capacity for the current configuration (spectral eff)
            cur_ch_cap = cal_RATE_one_sample_one_channel(cur_ch, tx_power[:, :, j], noise)  # (num_config, num_D2D + 1)
            
            # Calculate the interference from D2D to CUE for the current power settings
            inter = cal_CUE_INTER_one_sample_one_channel(cur_ch, tx_power[:, :, j])  # (num_config, num_D2D + 1)
            
            # Update the total capacity
            cur_cap += cur_ch_cap  # (num_config, num_D2D + 1)
            
            # Check if the CUE interference (caused by all other D2D users) is below the threshold
            CUE_mask *= (inter[:, num_D2D_user] < I_thr)  # (num_config,)

        # Check if D2D users meet the transmission requirement
        for j in range(num_D2D_user):
            DUE_mask *= (cur_cap[:, j] > DUE_thr)  
        # all users in all configurations should meet the transmission requirements
        # shape of Due mask = # (num_config,)

        # Expand the dimensions of masks for broadcasting
        CUE_mask = np.expand_dims(CUE_mask, -1)  # (num_config, 1)
        DUE_mask = np.expand_dims(DUE_mask, -1)  # (num_config, 1)

        # Calculate the total D2D Spectral Efficiency and Energy Efficiency
        sum_D2D_SE_temp = np.expand_dims(np.sum(cur_cap[:, :-1], axis=1), -1)  # (num_config, 1)
        sum_D2D_EE_temp = np.expand_dims(np.sum(cur_cap[:, :-1] / (np.sum(tx_power[:, :-1, :], axis=2) + P_c), axis=1), -1)   # (num_config, 1)

        # Filter results based on DUE mask
        D2D_SE_sum = sum_D2D_SE_temp * DUE_mask
        D2D_EE_sum = sum_D2D_EE_temp * DUE_mask

        # Select the configuration that maximizes either SE or EE
        if opt == "SE":
            arg_max_val = np.argmax(D2D_SE_sum)  # Index of the best SE
        else:
            arg_max_val = np.argmax(D2D_EE_sum)  # Index of the best EE

        max_SE = np.max(D2D_SE_sum)  # Maximum SE for this sample

        # Get the transmission power for the best configuration
        found_tx_val = tx_power[arg_max_val][:-1]  # Exclude CUE power

        # Store the optimal power settings
        power_mat_SE.append(found_tx_val)

        # Collect channels that were infeasible (max SE is zero)
        if max_SE == 0:
            chan_infea_mat.append(channel[i])

    # Convert list of powers to a numpy array
    power_mat_SE = np.array(power_mat_SE)

    # Calculate total SE, EE, and violation probabilities using the optimal power settings
    tot_SE, tot_EE, PRO_CUE_vio, PRO_DUE_vio = cal_rate_NP(channel, power_mat_SE, tx_max, noise, DUE_thr, I_thr, P_c)

    return tot_SE, tot_EE, PRO_CUE_vio, PRO_DUE_vio, np.array(chan_infea_mat)



## Calculate data rate for single channel, single sample
def cal_RATE_one_sample_one_channel(channel, tx_power, noise):
    # CHANN SHAPE = (NUM_USERS, NUM_USERS)
    diag_ch = np.diag(channel)
    inter_ch = channel-np.diag(diag_ch)
    tot_ch = np.multiply(channel, np.expand_dims(tx_power, -1))  # (num_config, num_D2D + 1, num_D2D + 1)
    int_ch = np.multiply(inter_ch, np.expand_dims(tx_power, -1)) # (num_config, num_D2D + 1, num_D2D + 1)
    sig_ch = np.sum(tot_ch-int_ch, axis=1)    # (num_config, num_D2D + 1)
    int_ch = np.sum(int_ch, axis=1)    # (num_config, num_D2D + 1)
    SINR_val = np.divide(sig_ch, int_ch+noise)
    cap_val = np.log2(1.0+SINR_val)
    return cap_val  # (num_config, num_D2D + 1)


def cal_CUE_INTER_one_sample_one_channel(channel, tx_power):
    # tx_power = (num_config, num_d2d +1, 1)
    diag_ch = np.diag(channel)
    inter_ch = channel-np.diag(diag_ch)  # (num_d2d +1, num_d2d+1)
    int_ch = np.multiply(inter_ch, np.expand_dims(tx_power, -1)) # (num_config, num_D2D + 1, num_D2D + 1)
    int_ch = np.sum(int_ch, axis=1)  # sum across all users (num_config, num_D2D + 1)
    return int_ch   # (num_config, num_D2D + 1)




def optimal_power_w_chan(channel, tx_max, noise, DUE_thr, I_thr, P_c, tx_power_set, opt="SE"):
    '''
    This function calculates the optimal transmission power for Device-to-Device (D2D) communications,
    taking into account the specific channel conditions. The function aims to maximize either 
    Spectral Efficiency (SE) or Energy Efficiency (EE) while ensuring constraints are met.

    Parameters:
    - channel: A 3D array representing the channel gains for each user and sample.
    - tx_max: Maximum transmission power allowed for each user.
    - noise: Noise level affecting the transmission.
    - DUE_thr: Minimum capacity threshold for D2D users.
    - I_thr: Interference threshold for CUE (Cellular User Equipment).
    - P_c: Constant power consumed regardless of transmission.
    - tx_power_set: Set of transmission power configurations to consider.
    - opt: Objective for optimization, either "SE" for Spectral Efficiency or "EE" for Energy Efficiency.

    Returns:
    - tot_SE: Total Spectral Efficiency.
    - tot_EE: Total Energy Efficiency.
    - PRO_CUE_vio: Probability of CUE violations.
    - PRO_DUE_vio: Probability of D2D violations.
    - chan_infea_mat: Channels that were found infeasible.
    - power_mat_SE: Array of optimal power settings for each sample.
    - channel: The input channel array for reference.
    '''

    # Extract the number of channels, D2D users, and samples from the channel data
    num_samples = channel.shape[0]
    num_channel = channel.shape[1]
    num_D2D_user = channel.shape[2] - 1  # Last user is the CUE
    
    # Initialize total Spectral Efficiency and lists to store power settings and infeasible channels
    tot_SE = 0
    power_mat_SE = []
    chan_infea_mat = []

    # Iterate through each sample to calculate optimal power
    for i in range(num_samples):
        cur_cap = 0  # Current capacity
        DUE_mask = 1  # Mask to check if DUE threshold is met
        CUE_mask = 1  # Mask to check if CUE interference is acceptable
        
        # Prepare the transmission power for the current configuration
        # Append a zero power for the CUE (last user) since we're focusing on D2D users
        tx_power = tx_max * np.hstack((tx_power_set, 0 * np.ones((tx_power_set.shape[0], 1, num_channel))))

        # Loop through each channel to calculate capacity
        for j in range(num_channel):
            cur_ch = channel[i][j]  # Current channel matrix for this sample and channel
            
            # Calculate the channel capacity for the current configuration
            cur_ch_cap = cal_RATE_one_sample_one_channel(cur_ch, tx_power[:, :, j], noise)
            
            # Calculate the interference from D2D to CUE for the current power settings
            inter = cal_CUE_INTER_one_sample_one_channel(cur_ch, tx_power[:, :, j])
            
            # Update the total capacity
            cur_cap += cur_ch_cap
            
            # Check if the CUE interference is below the threshold
            CUE_mask *= (inter[:, num_D2D_user] < I_thr)

        # Check if D2D users meet the transmission requirement
        for j in range(num_D2D_user):
            DUE_mask *= (cur_cap[:, j] > DUE_thr)

        # Expand the dimensions of masks for broadcasting
        CUE_mask = np.expand_dims(CUE_mask, -1)
        DUE_mask = np.expand_dims(DUE_mask, -1)

        # Calculate the total D2D Spectral Efficiency and Energy Efficiency
        sum_D2D_SE_temp = np.expand_dims(np.sum(cur_cap[:, :-1], axis=1), -1)
        sum_D2D_EE_temp = np.expand_dims(np.sum(cur_cap[:, :-1] / (np.sum(tx_power[:, :-1, :], axis=2) + P_c), axis=1), -1)

        # Store results for SE and EE calculations
        D2D_SE_sum = sum_D2D_SE_temp
        D2D_EE_sum = sum_D2D_EE_temp

        # Select the configuration that maximizes either SE or EE
        if opt == "SE":
            arg_max_val = np.argmax(D2D_SE_sum)  # Index of the best SE
        else:
            arg_max_val = np.argmax(D2D_EE_sum)  # Index of the best EE

        max_SE = np.max(D2D_SE_sum)  # Maximum SE for this sample

        # Get the transmission power for the best configuration
        found_tx_val = tx_power[arg_max_val][:-1]  # Exclude CUE power

        # Store the optimal power settings
        power_mat_SE.append(found_tx_val)

        # Collect channels that were infeasible (max SE is zero)
        if max_SE == 0:
            chan_infea_mat.append(channel[i])

    # Convert list of powers to a numpy array
    power_mat_SE = np.array(power_mat_SE)

    # Calculate total SE, EE, and violation probabilities using the optimal power settings
    tot_SE, tot_EE, PRO_CUE_vio, PRO_DUE_vio = cal_rate_NP(channel, power_mat_SE, tx_max, noise, DUE_thr, I_thr, P_c)

    return tot_SE, tot_EE, PRO_CUE_vio, PRO_DUE_vio, np.array(chan_infea_mat), np.array(power_mat_SE), np.array(channel)


def cal_SE_EE(channel, tx_max, noise, DUE_thr, I_thr, P_c, tx_power_mat, opt="SE"):
    num_channel = 1
    num_D2D_user = channel.shape[0] - 1
    tot_SE = 0

    cur_cap = 0
    DUE_mask = 1
    CUE_mask = 1

    tx_power = np.vstack((tx_power_mat, 0 * np.ones((1, 1))))
    tx_power = np.expand_dims(tx_power, 0)

    cur_ch = channel
    cur_ch_cap = cal_RATE_one_sample_one_channel(cur_ch, tx_power[:, :, 0], noise)
    cur_cap = cur_cap + cur_ch_cap


    sum_D2D_SE_temp = np.sum(cur_cap[0,:-1])
    sum_D2D_EE_temp = np.sum(cur_cap[0,:-1] / (tx_power[0,:-1, 0] + P_c))

    D2D_SE_sum = sum_D2D_SE_temp
    D2D_EE_sum = sum_D2D_EE_temp

    return D2D_SE_sum, D2D_EE_sum



np.random.seed(0)

Num_power_level = 100
tx_power_set = all_possible_tx_power(Num_channel, Num_user, Num_power_level - 1)

Size_area = 20
D2D_dist = 15
tx_max = 10**2.0

DUE_thr = 4.0
I_thr = 10**(-55.0/10)
P_c = 2*10**2.0
BW = 1e7
noise = BW*10**-17.4

llm = Llama(
      #model_path="./models/codellama-7b.Q5_K_M.gguf",
      #model_path="./models/codellama-7b.Q4_K_M.gguf",
      model_path="./models/llama-2-13b.Q5_K_M.gguf",
      #n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      n_ctx=26200,
      verbose=False # Uncomment to increase the context window
)

query_text = """
Take a deep breath and work on this problem step-by-step. You are a mathematical tool to predict some model. Your job is to predict B for given A. The following is the dataset that you can use for the prediction.
If A is 59.6, 73.2, 59.8, 63.6, then B is 100, 100.
If A is 71.0, 59.8, 61.9, 73.7, then B is 0, 100.
If A is 61.4, 65.8, 66.0, 66.9, then B is 100, 0.
If A is 62.3, 58.9, 72.8, 54.0, then B is 100, 100.
If A is 48.6, 55.0, 52.0, 48.9, then B is 100, 100.
If A is 74.3, 57.9, 76.7, 62.9, then B is 0, 100.
If A is 53.4, 51.3, 61.1, 68.9, then B is 100, 0.
If A is 83.0, 55.9, 68.0, 56.6, then B is 0, 100.
If A is 60.6, 65.0, 66.7, 58.6, then B is 100, 100.
If A is 72.1, 69.6, 58.7, 54.3, then B is 0, 100.
If A is 58.7, 71.6, 72.1, 50.1, then B is 100, 100.
If A is 77.4, 53.2, 60.8, 70.7, then B is 0, 100.
If A is 59.4, 80.9, 66.6, 63.4, then B is 100, 100.
If A is 63.7, 51.3, 59.7, 65.9, then B is 0, 100.
If A is 63.1, 68.3, 84.8, 63.8, then B is 100, 100.
If A is 74.1, 62.3, 64.0, 68.9, then B is 0, 100.
If A is 70.1, 75.8, 50.0, 70.0, then B is 100, 0.
If A is 60.3, 62.0, 64.2, 74.0, then B is 100, 0.
If A is 52.1, 64.6, 57.4, 52.2, then B is 0, 100.
If A is 67.4, 68.2, 55.5, 70.0, then B is 0, 100.
If A is 85.8, 64.6, 59.1, 70.4, then B is 0, 100.
If A is 55.9, 59.1, 63.3, 52.2, then B is 0, 100.
If A is 61.4, 65.9, 68.4, 64.5, then B is 0, 100.
If A is 55.7, 59.2, 62.2, 54.0, then B is 0, 100.
If A is 49.4, 63.1, 50.3, 62.9, then B is 100, 0.
If A is 52.1, 71.5, 58.6, 60.1, then B is 100, 100.
If A is 71.1, 54.9, 51.3, 57.8, then B is 0, 100.
If A is 66.0, 60.1, 73.5, 69.6, then B is 100, 100.
If A is 58.7, 48.0, 63.0, 53.3, then B is 0, 100.
If A is 71.5, 64.3, 65.2, 67.5, then B is 0, 100.
If A is 67.3, 67.0, 78.9, 67.3, then B is 0, 100.
If A is 62.1, 59.9, 54.4, 75.7, then B is 100, 0.
If A is 76.7, 50.1, 82.5, 52.4, then B is 0, 100.
If A is 58.8, 66.3, 64.6, 68.1, then B is 100, 0.
If A is 64.6, 61.3, 49.5, 55.8, then B is 100, 0.
If A is 62.5, 64.8, 66.2, 51.7, then B is 0, 100.
If A is 67.6, 57.1, 69.1, 75.8, then B is 100, 0.
If A is 74.5, 57.0, 72.4, 61.1, then B is 100, 0.
If A is 66.5, 71.0, 69.4, 66.3, then B is 100, 0.
If A is 55.4, 55.7, 54.4, 55.5, then B is 100, 0.
If A is 67.2, 70.0, 73.2, 76.8, then B is 100, 100.
If A is 58.4, 59.6, 58.5, 50.8, then B is 100, 0.
If A is 68.7, 48.6, 67.5, 63.0, then B is 0, 100.
If A is 75.0, 57.6, 50.3, 65.5, then B is 0, 100.
If A is 52.6, 60.2, 69.6, 55.6, then B is 100, 0.
If A is 64.3, 69.1, 70.2, 77.7, then B is 100, 100.
If A is 51.9, 62.6, 111.9, 68.7, then B is 100, 100.
If A is 61.2, 79.7, 58.8, 66.1, then B is 100, 0.
If A is 63.2, 61.8, 54.0, 69.1, then B is 100, 0.
If A is 74.4, 68.0, 71.1, 60.3, then B is 0, 100.
If A is 61.7, 66.9, 64.9, 60.1, then B is 0, 100.
If A is 84.6, 51.6, 62.6, 54.9, then B is 0, 100.
If A is 54.2, 53.2, 49.7, 53.7, then B is 100, 0.
If A is 55.9, 50.2, 70.5, 85.1, then B is 100, 0.
If A is 69.2, 67.5, 52.5, 71.9, then B is 100, 100.
If A is 58.6, 66.6, 49.8, 65.4, then B is 100, 0.
If A is 77.2, 75.1, 74.2, 69.9, then B is 100, 100.
If A is 79.5, 59.9, 67.1, 63.3, then B is 0, 100.
If A is 61.7, 63.5, 66.6, 82.9, then B is 100, 0.
If A is 66.6, 64.2, 67.7, 67.3, then B is 100, 0.
If A is 73.7, 57.0, 65.8, 54.3, then B is 0, 100.
If A is 56.2, 62.9, 50.8, 66.1, then B is 100, 0.
If A is 57.9, 57.3, 53.2, 47.5, then B is 0, 100.
If A is 64.3, 67.8, 60.9, 55.0, then B is 100, 0.
If A is 72.1, 56.2, 56.2, 69.5, then B is 100, 0.
If A is 52.4, 67.5, 57.1, 55.3, then B is 
"""

##  100 and F is 100.



Num_sample = 100000
ch_mat, rx_mat, tx_mat, CUE_mat = ch_gen(Size_area, D2D_dist, Num_user, Num_channel, Num_sample)
ch_mat_log = np.log(ch_mat)
chan_avg = np.mean(ch_mat_log)
chan_std = np.std(ch_mat_log)

##################
### Original setting
##################
Size_area = 20
D2D_dist = 15
tx_max = 100
P_c = 2*10**2.0
##################


Num_sample = 1000


print(llm(query_text, stop=["."])["choices"][0]["text"])

batch_size = 100
critera = "EE"

P_c = 5*10**2.0
Size_area = 70
D2D_dist = 20

ch_mat_val, rx_mat_val, tx_mat_val, CUE_mat_val = ch_gen(Size_area, D2D_dist, Num_user, Num_channel, 1000)
SE_OPT_val, EE_OPT_val, CUE_vio_OPT_val, DUE_vio_OPT, INF_CHAN_MAT_val, PW_VEC_val, CHAN_VEC_val = optimal_power_w_chan(ch_mat_val, tx_max, noise,
                                                                                                DUE_thr, I_thr, P_c,
                                                                                                tx_power_set, opt=critera)

for i_1 in range(5):

    batch_size = 25*(2**(i_1))
    batch_size = 800
    #batch_size = 200
    print("batch_size = ", batch_size)
    SE_opt_mat = 0
    EE_opt_mat = 0

    SE_prop_mat = 0
    EE_prop_mat = 0

    SE_prop_2_mat = 0
    EE_prop_2_mat = 0


    SE_rand_mat = 0
    EE_rand_mat = 0


    SE_bin_mat = 0
    EE_bin_mat = 0


    for j in range(501):

        ch_mat, rx_mat, tx_mat, CUE_mat = ch_gen(Size_area, D2D_dist, Num_user, Num_channel, Num_sample)
        SE_OPT, EE_OPT, CUE_vio_OPT, DUE_vio_OPT, INF_CHAN_MAT, PW_VEC, CHAN_VEC = optimal_power_w_chan(ch_mat, tx_max, noise, DUE_thr, I_thr, P_c, tx_power_set, opt=critera)
        query_text = 'Take a deep breath and work on this problem step-by-step. You are a mathematical tool to predict some model. Your job is to predict B for given A. The following is the dataset that you can use for the prediction.\n'
        for i in range(PW_VEC.shape[0]):
            chan_revised = (np.log(ch_mat[i, 0, :, :]) - chan_avg) / chan_std * 100

            if i == PW_VEC.shape[0]-1:
                chan_revised_val = (np.log(ch_mat_val[j, 0, :, :]) - chan_avg) / chan_std * 100
                query_text = query_text + f'If A is {chan_revised_val[0, 0]:0.0f}, {chan_revised_val[0, 1]:0.0f}, {chan_revised_val[1, 0]:0.0f}, {chan_revised_val[1, 1]:0.0f}, then B is '
                #print(f'[TRUE VALUE] If A is {chan_revised[0, 0]:0.2f}, {chan_revised[0, 1]:0.0f}, {chan_revised[1, 0]:0.0f}, {chan_revised[1, 1]:0.0f}, then B is ')
                #print(f'[TRUE VALUE] B is {PW_VEC[i, 0, 0]:0.0f}, {PW_VEC[i, 1, 0]:0.0f}')

                SE_opt, EE_opt = cal_SE_EE(ch_mat_val[i, 0, :, :], tx_max, noise, DUE_thr, I_thr, P_c, PW_VEC_val[j], opt=critera)
                #print("SE_opt = ", SE_opt, "EE_opt = ", EE_opt*1000)

            if i < batch_size:
                query_text = query_text + f'If A is {chan_revised[0, 0]:0.0f}, {chan_revised[0, 1]:0.0f}, {chan_revised[1, 0]:0.0f}, {chan_revised[1, 1]:0.0f}, then B is {PW_VEC[i, 0, 0]:0.0f}, {PW_VEC[i, 1, 0]:0.0f}.\n'
        llm_result = llm(query_text, stop=["."])["choices"][0]["text"]
        #print("qurert_text = ", query_text)
        #print("LLM RESULT = ", llm_result)
        SE_prop, EE_prop = 0, 0
        temp_dict = llm_result.split(",")
        if len(temp_dict) == 2:
            try:
                temp_PW = np.expand_dims(np.asarray(temp_dict).astype(float), -1)
            except ValueError:
                temp_PW = 0 * np.random.rand(2, 1)
            SE_prop, EE_prop = cal_SE_EE(ch_mat_val[j, 0, :, :], tx_max, noise, DUE_thr, I_thr, P_c, temp_PW, opt=critera)
            #print("SE_prop = ", SE_prop, "EE_prop = ", EE_prop * 1000)

        temp_PW_rand = tx_max * np.random.rand(2, 1)

        #print("temp_PW_rand = ", temp_PW_rand)
        SE_rand, EE_rand = cal_SE_EE(ch_mat_val[j, 0, :, :], tx_max, noise, DUE_thr, I_thr, P_c, temp_PW_rand, opt=critera)
        #print("SE_rand = ", SE_rand, "EE_rand = ", EE_rand * 1000)
        #print("**"*50)

        temp_val = np.random.rand()
        if temp_val < 0.5:
            temp_PW_rand[0, 0] = 100
            temp_PW_rand[1, 0] = 0
        else:
            temp_PW_rand[1, 0] = 100
            temp_PW_rand[0, 0] = 0

        #print("temp_PW_rand = ", temp_PW_rand)
        SE_bin, EE_bin = cal_SE_EE(ch_mat_val[j, 0, :, :], tx_max, noise, DUE_thr, I_thr, P_c, temp_PW_rand, opt="EE")
        #print("SE_rand = ", SE_rand, "EE_rand = ", EE_rand * 1000)
        #print("**"*50)
        SE_OPT, EE_OPT = cal_SE_EE(ch_mat_val[j, 0, :, :], tx_max, noise, DUE_thr, I_thr, P_c, PW_VEC_val[j], opt=critera)

        if critera == "SE":
            if SE_bin > SE_prop:
                SE_prop_2 = SE_bin
                EE_prop_2 = EE_bin
            else:
                SE_prop_2 = SE_prop
                EE_prop_2 = EE_prop

        if critera == "EE":
            if EE_bin > EE_prop:
                SE_prop_2 = SE_bin
                EE_prop_2 = EE_bin
            else:
                SE_prop_2 = SE_prop
                EE_prop_2 = EE_prop



        SE_opt_mat = SE_opt_mat + SE_OPT
        EE_opt_mat = EE_opt_mat + EE_OPT*1000

        SE_prop_mat = SE_prop_mat + SE_prop
        EE_prop_mat = EE_prop_mat + EE_prop*1000

        SE_prop_2_mat = SE_prop_2_mat + SE_prop_2
        EE_prop_2_mat = EE_prop_2_mat + EE_prop_2*1000


        SE_rand_mat = SE_rand_mat + SE_rand
        EE_rand_mat = EE_rand_mat + EE_rand*1000

        SE_bin_mat = SE_bin_mat + SE_bin
        EE_bin_mat = EE_bin_mat + EE_bin*1000


        if j%50 == 0:
            print(print(f'index = {j+1}: [OPT] SE: {SE_opt_mat/(j+1):0.1f}, EE: {EE_opt_mat/(j+1):0.1f}, [PROP] SE: {SE_prop_mat/(j+1):0.1f}, EE: {EE_prop_mat/(j+1):0.1f}, [PROP_2] SE: {SE_prop_2_mat/(j+1):0.1f}, EE: {EE_prop_2_mat/(j+1):0.1f}, [RAND] SE: {SE_rand_mat/(j+1):0.1f}, EE: {EE_rand_mat/(j+1):0.1f} , [bin] SE: {SE_bin_mat/(j+1):0.1f}, EE: {EE_bin_mat/(j+1):0.1f}'))

    print("Final results")
    print(f'batch_size = {batch_size}: [OPT] SE: {SE_opt_mat / (j + 1):0.1f}, EE: {EE_opt_mat / (j + 1):0.1f}, [PROP] SE: {SE_prop_mat / (j + 1):0.1f}, EE: {EE_prop_mat / (j + 1):0.1f}, [RAND] SE: {SE_rand_mat / (j + 1):0.1f}, EE: {EE_rand_mat / (j + 1):0.1f}')

    print("*" * 50)