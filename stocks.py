import pandas as pd
import os
import numpy as np
import torch
import pandas as pd
#start_ = -1
#end_ = 1e12
def get_stock_data(start_, end_, stock, seq_length):

    # pth = r"C:\Users\rdadf\work\fnmaster\dataset"
    # #stock = 'AMZN'
    # lob_level = 1
    # p = 'lobdataread'
    # msgfile = os.path.join(pth, f'{p}' ,"data", f'{stock}_2012-06-21_34200000_57600000_message_{lob_level}.csv')
    # obfile =  os.path.join(pth, f'{p}' ,"data", f'{stock}_2012-06-21_34200000_57600000_orderbook_{lob_level}.csv')
    # df1=pd.read_csv(msgfile, sep=',', header=None, 
    #         names=['time', 'type', 'ord_id', 'size', 'price', 'dir'],
    #         dtype={'time':np.float64, 'type':np.int32, 'ord_id':np.int32, 
    #             'size':np.int64, ',':np.int64, 'dir':np.int32})

    # df2=pd.read_csv(obfile, sep=',', header=None, 
    #         names=['ask_prc_L1', 'ask_sz_L1', 'bid_prc_L1', 'bid_sz_L1'],
    #         dtype={'ask_prc_L1':np.float64, 'ask_sz_L1':np.int64, 'bid_prc_L1':np.float64, 'bid_sz_L1':np.int64})

    # df3 = pd.concat([df1, df2], axis=1)
    # df3 = df3.drop_duplicates(subset='time', keep='first')
    # df3['mid_price'] = (df3['ask_prc_L1'] + df3['bid_prc_L1'])/2
    # df3['datetime'] = pd.to_timedelta(df3['time'], unit='s') + pd.Timestamp("2012-06-21")
    # df3 = df3[(df3['datetime'].dt.time >= pd.Timestamp('2012-06-21 09:46:00').time()) & 
    #                 (df3['datetime'].dt.time <= pd.Timestamp('2012-06-21 15:44:00').time())]
    # df = df3[['time', 'datetime', 'mid_price']].copy()
    # df['mid_price'] = df['mid_price']/10000 
    # df.loc[:, 'mid_price'] = df['mid_price'] - df['mid_price'].iloc[0]
    # df.loc[:, 'time'] = df['time'] - df['time'].iloc[0]
    # df.reset_index(drop=True, inplace=True)
    # df['mid_diff'] = df['mid_price'].diff()
    # # Replace NaN with 0 at the beginning
    # df['mid_diff'].fillna(0, inplace=True)
    # threshold = 1e-6
    # df = df[df['mid_diff'].abs() >= threshold]
    # df.reset_index(drop=True, inplace=True)
    # df_positive = df[df['mid_diff'] >= 0]
    # df_negative = df[df['mid_diff'] < 0]
    # quantiles_positive = df_positive['mid_diff'].quantile(0.5)
    # quantiles_negative = df_negative['mid_diff'].quantile(0.5)
    # quantiles_positive, quantiles_negative
    # df['mid_diff'] = np.where(df['mid_diff'] >= 0, 0.01, -0.01)
    # df = df[['time', 'datetime', 'mid_diff', 'mid_price']]
    # df['state'] = np.nan
    # df.loc[df['mid_diff'] < 0, 'state'] = 0
    # df.loc[df['mid_diff'] >= 0, 'state'] = 1 
    # df['state'] = df['state'].astype(int)

    df = pd.read_csv('data/AMZN.csv')
    df = df[((df['time'] >= start_) & (df['time'] <= end_))]
    #seq_length = 250
    # Divide the total time into 250 periods
    total_time = df['time'].iloc[-1]
    period_length = total_time / seq_length
    periods = []

    # Create events for each period

    for i in range(seq_length):
        start_time = i * period_length
        end_time = start_time + period_length
        period_events = []
        period_df = df[(df['time'] >= start_time) & (df['time'] < end_time)]
        #last_event_times = {}
        
        global_idx_event = 0
        period_df.iloc[1, -1] = 1
        for counter, (idx, row) in enumerate(period_df.iterrows()):
            event = {}
            global_idx_event += 1
            event['idx_event'] = global_idx_event
            
            if counter > 0:
                event['time_since_last_event'] = row['time'] - period_df['time'].iloc[counter-1]
            else:
                event['time_since_last_event'] = row['time'] - start_time
            
            same_event_times = period_df[period_df['state'] == row['state']]['time']
            if same_event_times.iloc[0] != row['time']:
                event['time_since_last_same_event'] = row['time'] - same_event_times.iloc[same_event_times.index.get_loc(idx)-1]
            else:
                event['time_since_last_same_event'] = row['time']  - start_time
            
            event['type_event'] = row['state']
            event['time_since_start'] = row['time']
            period_events.append(event)
        
        periods.append(period_events)
    data = periods
    time_durations = []
    type_seqs = []
    seq_lens = []
    for i in range(len(data)):
        seq_lens.append(len(data[i]))
        type_seqs.append(torch.LongTensor([int(event['type_event']) for event in data[i]]))
        time_durations.append(torch.FloatTensor([float(event['time_since_last_event']) for event in data[i]]))
    
    return time_durations, type_seqs, seq_lens, data
