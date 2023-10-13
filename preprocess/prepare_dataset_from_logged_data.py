import pandas as pd

data = []

for pid in [2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]:
    plays = pd.read_csv(f'../../../personalization/data/{pid}/plays.csv')
    searches = pd.read_csv(f'../../../personalization/data/{pid}/searches.csv')
    choices = pd.read_csv(f'../../../personalization/data/{pid}/choices.csv')

    search_data = []
    choice_data = []
    not_found = []

    for modality in [('visual', 'Video'), ('auditory', 'Audio'), ('kinesthetic', 'Movement')]:
        for signal in ['idle',  'searching', 'has_item', 'has_information']:
            
            try:
                p = plays.query(f'type == "{modality[0]}" and signal == "{signal}"')
                s = searches.query(f'type == "{modality[1]}" and signal == "{signal}"').dropna()[['results', 'time']].values
                c = choices.query(f'type == "{modality[0]}" and signal == "{signal}"').dropna()[['query', 'time']].values
                
                # print(q, c)

                for _, row in p.iterrows():
                    time, id, signal, stim_type = row['time'], row['id'], row['signal'], row['type']
                    found = False

                    for results, result_time in s:
                        if id in results.split(','): # and time > result_time - 25:
                            search_data.append(( stim_type, signal, id, results))
                            found = True
                            break

                    for choice, choice_time in c:
                        if id in choice.split(',') and time > choice_time - 25:
                            choice_data.append((stim_type, signal, id, choice))
                            found = True
                            break
                            
                    if not found:
                        # print(f'couldnt find {id}')
                        not_found.append((stim_type, id))
            
            except Exception as e:
                print(f'error for {modality} {signal} {pid}: {e} ')
            
    
    print(pid, len(search_data), len(choice_data), len(not_found))

    for stim_type, signal, id, results in search_data:
        selected = [row[2] for row in search_data if row[3] == results]
        unselected = [id for id in results.split(',') if id not in selected]

        if len(unselected) > 0:
            data.append({
                                    'pid': pid,
                                    'type': stim_type,
                                    'signal': signal,
                                    'chosen': ','.join(str(candidate) for candidate in set(selected)),
                                    'options': ','.join(str(candidate) for candidate in set(unselected)),
                                })
        
    for stim_type, signal, id, results in choice_data:
        selected = [row[2] for row in choice_data if row[3] == results]
        unselected = [id for id in results.split(',') if id not in selected]

        if len(unselected) > 0:
            data.append({
                                    'pid': pid,
                                    'type': stim_type,
                                    'signal': signal,
                                    'chosen': ','.join(str(candidate) for candidate in set(selected)),
                                    'options': ','.join(str(candidate) for candidate in set(unselected)),
                                })

            
            # if len(q) > 1:
            #     selected_pool = []

                # q_index = 0
                # candidate_pool = q.iloc[q_index]['results'] #pop top of query df
                # q_index+=1

                # c_index = 0
                # choices_pool = c.iloc[c_index]['query'] #pop top of choice df
                # c_index += 1

                # for _ , row in p.iterrows():
                #     time, id, signal, stim_type = row['time'], row['id'], row['signal'], row['type']

                #     #normal case
                #     if q.iloc[q_index]['time'] > time:
                #         if int(id) > -1:
                #             selected_pool.append(id)

                #             if id not in candidate_pool.split(',') and \
                #                     id not in choices_pool and \
                #                     c.iloc[c_index]['time'] <= time and c_index < len(c) - 1:
                                
                #                 while c.iloc[c_index]['time'] <= time and c_index < len(c) - 1:
                #                     choices_pool = c.iloc[c_index]['query'] #pop
                #                     c_index += 1
                #                     if id in choices_pool:
                #                         break

                                

                #             if id in candidate_pool.split(','):
                #                 print(f'{id} from candidates')
                #             elif id in choices_pool:
                #                 print(f'{id} from query')
                #             else:
                #                 print(f'{id} from ????UNKNOWN???')


                #     #switch to next query set
                #     elif q.iloc[q_index]['time'] <= time:
                #         # print('switch!')
                #         # print(selected_pool, candidate_pool.split(','))

                #         #log the row
                #         if len(selected_pool) > 0:
                #             data.append({
                #                 'pid': pid,
                #                 'type': stim_type,
                #                 'signal': signal,
                #                 'chosen': ','.join(str(candidate) for candidate in selected_pool),
                #                 'options': candidate_pool,
                #             })


                #         #end case
                #         if q_index < len(q) - 2:
                #             candidate_pool = q.iloc[q_index]['results'] #pop
                #             q_index += 1
                            
                #         else:
                #             candidate_pool = ""

                #         selected_pool = []
                
                # # print(selected_pool, candidate_pool.split(','))
                # if len(selected_pool) > 0:
                #     #log the final row
                #     data.append({
                #                 'pid': pid,
                #                 'type': stim_type,
                #                 'signal': signal,
                #                 'chosen': ','.join(str(candidate) for candidate in selected_pool),
                #                 'options': candidate_pool,
                                
                #             })
                
                


df = pd.DataFrame(data)
# df = df.drop_duplicates()
# df = df.dropna()
df.to_csv('../data/plays_and_options.csv')




