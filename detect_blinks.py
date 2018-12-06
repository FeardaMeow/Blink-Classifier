from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import make_scorer, recall_score, precision_score, accuracy_score
import pandas as pd
from scipy.stats import uniform, randint
from collections import namedtuple
import warnings
import pickle
import glob

def eye_aspect_ratio(eye):
    #Vertical eye euclidean distance
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    #Horizontal eye euclidean distance
    C = dist.euclidean(eye[0], eye[3])
    
    #comput the eye aspect ratio
    ear = (A + B)/(2.0 * C)
    
    return(ear)

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True,
#	help="path to facial landmark predictor")
#ap.add_argument("-v", "--video", type=str, default="",
#	help="path to input video file")
#args = vars(ap.parse_args())

def annotate_video(video):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    vs = cv2.VideoCapture(video)
    while not vs.isOpened():
        vs = cv2.VideoCapture(video)
        cv2.waitKey(100)
        print("Wait for the header")
    
    num_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)-30
    frame_annotate = np.zeros(int(num_frames))
    
    current_frame = vs.get(cv2.CAP_PROP_POS_FRAMES)
    
    while True:
        if current_frame%150 == 0:
            print(current_frame)
        flag, frame = vs.read()
        #Checks to see if there are any more frames to read
        if flag:
            frame = frame[540:,350:960,:]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale frame
            rects = detector(gray, 0)
            
            for rect in rects:
        		# determine the facial landmarks for the face region, then
        		# convert the facial landmark (x, y)-coordinates to a NumPy
        		# array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
          
        		# extract the left and right eye coordinates, then use the
        		# coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
          
        		# average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0
                frame_annotate[int(current_frame)] = ear
            current_frame = vs.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            vs.set(cv2.CAP_PROP_POS_FRAMES, current_frame-1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)
        
        if cv2.waitKey(10) == 27:
            break
        if vs.get(cv2.CAP_PROP_POS_FRAMES) == num_frames:
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break
        
    vs.release()
    np.savetxt(video.split('.')[0]+"_EAR.csv", frame_annotate, delimiter=",")

def frame_checker(frame_list, video):
    vs = cv2.VideoCapture(video)
    while not vs.isOpened():
        vs = cv2.VideoCapture(video)
        cv2.waitKey(10)
        print("Wait for the header")
        
    for i in frame_list:
        vs.set(cv2.CAP_PROP_POS_FRAMES, i)
        flag, frame = vs.read()
        if flag:
            frame = imutils.resize(frame[540:,350:960,:])
            cv2.imshow("Frame", frame)
            cv2.waitKey(0)
        else:
            vs.set(cv2.CAP_PROP_POS_FRAMES, cv2.CAP_PROP_POS_FRAMES-1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)
        
    cv2.destroyAllWindows()
    vs.release()

def label_video(video):
    vs = cv2.VideoCapture(video)
    while not vs.isOpened():
        vs = cv2.VideoCapture(video)
        cv2.waitKey(10)
        print("Wait for the header")
    
    num_frames = vs.get(cv2.CAP_PROP_FRAME_COUNT)
    video_label = np.zeros(int(num_frames))
    current_frame = 0
    video_frame = []
    
    while True:
        flag, frame = vs.read()
        if flag:
            frame = frame[540:,350:960,:]
            video_frame.append(frame)
            cv2.putText(frame, "Frame: {}".format(current_frame), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(40)
        
            if key == ord('p'):
                frame_counter = len(video_frame) - 1
                frame_pause = video_frame[int(frame_counter)]
                while True:
                    key2 = cv2.waitKey(0)
                    if key2 == ord('b'):
                        video_label[current_frame - (len(video_frame) - frame_counter - 1)] = 1
                        cv2.putText(frame_pause, "Paused", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame_pause, "Labeled: {}".format(current_frame - (len(video_frame) - frame_counter-1)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("Frame", frame_pause)
                    elif key2 == ord('k'):
                        if frame_counter > 0:
                            frame_counter -= 1
                        frame_pause = video_frame[int(frame_counter)]
                        cv2.putText(frame_pause, "Paused".format(frame_counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("Frame", frame_pause)
                    elif key2 == ord('p'):
                        break
                    else:
                        if frame_counter < len(video_frame) - 1:
                            frame_counter += 1
                        frame_pause = video_frame[int(frame_counter)]
                        cv2.putText(frame_pause, "Paused".format(frame_counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("Frame", frame_pause)
            elif key == ord('q'):
                current_frame += 1
                break
                
            current_frame += 1
        else:
            vs.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            time.sleep(1.0)
        
        if current_frame >= 8*30:
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break
    
    while True:
        flag, frame = vs.read()
        if flag:
            frame = frame[540:,350:960,:]
            video_frame.append(frame)
            video_frame.pop(0)
            cv2.putText(frame, "Frame: {}".format(current_frame), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(40)

            if key == ord('p'):
                frame_counter = len(video_frame) - 1
                frame_pause = video_frame[int(frame_counter)]
                while True:
                    key2 = cv2.waitKey(0)
                    if key2 == ord('b'):
                        video_label[current_frame - (len(video_frame) - frame_counter - 1)] = 1
                        cv2.putText(frame_pause, "Paused", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame_pause, "Labeled: {}".format(current_frame - (len(video_frame) - frame_counter-1)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("Frame", frame_pause)
                    elif key2 == ord('k'):
                        if frame_counter > 0:
                            frame_counter -= 1
                        frame_pause = video_frame[int(frame_counter)]
                        cv2.putText(frame_pause, "Paused", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("Frame", frame_pause)
                    elif key2 == ord('p'):
                        break
                    else:
                        if frame_counter < len(video_frame) - 1:
                            frame_counter += 1
                        frame_pause = video_frame[int(frame_counter)]
                        cv2.putText(frame_pause, "Paused", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow("Frame", frame_pause)
            elif key == ord('q'):
                break
            
            current_frame += 1
        else:
            vs.set(cv2.CAP_PROP_POS_FRAMES, current_frame-1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            time.sleep(1.0)
        
        if current_frame == num_frames:
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break
        
    cv2.destroyAllWindows()
    vs.release()
    np.savetxt(video.split('.')[0]+"_labels.csv", video_label, delimiter=",")

def gen_features(x, win=7):
    n = x.shape[0]
    x_transform = np.hstack( x[i:i+win] for i in range(0,n-win+1) ).reshape(n-win+1,win)
    return(x_transform)

def neg_indices(x, win=7, space = 5):
    neg_index = np.ones(x.shape[0])
    for i in np.where(x==1)[0]:
        lower = max(0,i-int(win/2))
        upper = min(x.shape[0],i+int(win/2)+1)
        neg_index[lower:upper] = 0
    
    space_index = np.zeros(x.shape[0])
    space_index[::5] = 1
    
    neg_data = np.zeros(x.shape[0])
    neg_data[::5] = 1
    
    return(np.logical_and(neg_data==1, neg_index==1))
    
def extract_data(x, labels, win):
    neg_index = neg_indices(labels, win=win)
    train_x = np.concatenate((x[np.where(labels==1)[0],:],x[neg_index,:]))
    train_label = np.concatenate((np.ones(x[np.where(labels==1)[0],:].shape[0]), np.zeros(x[neg_index,:].shape[0])))
    
    return(train_x, train_label)

def append_to_df(df, score, method, win, pid):
    data_dict = {'model':[method],'window':[win],'pid':[pid],'mean_acc':[score]}
    return(df.append(pd.DataFrame.from_dict(data_dict)))
    
def append_score_to_df(df, model, mtype, pid, accuracy, recall, precision):
    data_dict = {'model':[model],'type':[mtype],'pid':[pid],'accuracy':[accuracy],'recall':[recall],'precision':[precision]}
    return(df.append(pd.DataFrame.from_dict(data_dict)))

def getScores(model, x, y):
    scoreResults = namedtuple("scoreResults", "accuracy precision recall")
    
    pred_values = model.predict(x)
    
    recall = np.mean(pred_values[y==1] == y[y==1])
    #In case there are zero predictions of blinks
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            precision = np.mean(pred_values[pred_values==1] == y[pred_values==1])
        except RuntimeWarning:
            precision = 0
    accuracy = np.mean(pred_values == y) 
    
    return(scoreResults(accuracy=accuracy, precision=precision, recall=recall))

def randomizeSearchWrapper(x, y, model, param, file_name, scorers, n, refit_score='precision_score'):
    n_iter_search = n
    
    try:
        with open(file_name, 'rb') as f:
            random_search = pickle.load(f)
    except FileNotFoundError:
        random_search = RandomizedSearchCV(model, param_distributions=param, refit=refit_score,
                                           n_iter=n_iter_search, cv=5, verbose=10, scoring=scorers)
        random_search.fit(x, y)
        with open(file_name, 'wb') as f:
            pickle.dump(random_search, f)
    
    return(getScores(random_search, x, y))

def getData(top_directory, window, split=0.8):
    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []
    for filename in glob.iglob(top_directory + '/x/*.csv', recursive=True):
        x_temp = gen_features(np.loadtxt(filename, delimiter=',')[:18300], win=window)
        x_train_index = int(x_temp.shape[0]*split)
        x_train_list.append(x_temp[:x_train_index,:])
        x_test_list.append(x_temp[x_train_index:,:])
        
    for filename in glob.iglob(top_directory + '/y/*.csv', recursive=True):    
        y_temp = np.loadtxt(filename, delimiter=',')[int(window/2):18300-int(window/2)]
        y_train_index = int(y_temp.shape[0]*split)
        y_train_list.append(y_temp[:y_train_index])
        y_test_list.append(y_temp[y_train_index:])
        
    return(x_train_list, y_train_list, x_test_list, y_test_list)
        
#Trim 1.avi at 18300, 3.avis 18200, 4.avi 18200, 2.avi 18370, 5.avi 18000
#video = "5.avi"
#frame_annotate = annotate_video(video)
#label_video(video)
def main():
    #Metric scores
    scorers = {'accuracy_score': make_scorer(accuracy_score),
               'precision_score': make_scorer(precision_score),
               'recall_score': make_scorer(recall_score)}
    
    #Parameters to random server
    param_dist_rbf = {'C':uniform(0.001,100), 'gamma':uniform(1e-01,10000)}
    param_dist_linear = {'C':uniform(0.001,500)}
    param_dist_gb = {'min_samples_split':randint(2,25), 'min_samples_leaf':randint(1,20), 
                     'max_depth':randint(1,8), 'max_features':[1,2,3], 
                     'learning_rate':uniform(0.001,0.099), 'n_estimators':randint(100,500),
                     'subsample':uniform(0.6,0.4)}
    
    n_iter_search = 20
    window = 13
    
    #SVM with RBF kernel
    svm_rbf = svm.SVC(kernel='rbf')
    #SVM linear
    svm_linear = svm.SVC(kernel='linear')
    #GradientBoosting
    gb = GradientBoostingClassifier()
    
    #Get train and test data
    x_train_list, y_train_list, x_test_list, y_test_list = getData('data', window=window)
    final_train_models_linear = []
    final_train_models_rbf = []
    final_train_models_gb = []
    for x_i, y_i in zip(x_train_list, y_train_list):
        x_features, y_features = extract_data(x_i, y_i, window)
            
        #Scaler to standerize data
        scaler = StandardScaler()
        scaler.fit(x_features)
        
        #Linear SVM
        random_search = RandomizedSearchCV(svm_linear, param_distributions=param_dist_linear, 
                                           refit='accuracy_score', n_iter=n_iter_search, 
                                           cv=5, verbose=10, scoring=scorers, n_jobs=-1)
        random_search.fit(scaler.transform(x_features), y_features)
        final_train_models_linear.append(random_search)
        
        #RBF SVM
        random_search = RandomizedSearchCV(svm_rbf, param_distributions=param_dist_rbf, 
                                           refit='accuracy_score', n_iter=n_iter_search, 
                                           cv=5, verbose=10, scoring=scorers, n_jobs=-1)
        random_search.fit(scaler.transform(x_features), y_features)
        final_train_models_rbf.append(random_search)
        
        #GradientBoosting
        random_search = RandomizedSearchCV(gb, param_distributions=param_dist_gb, 
                                           refit='accuracy_score', n_iter=n_iter_search, 
                                           cv=5, verbose=10, scoring=scorers, n_jobs=-1)
        random_search.fit(scaler.transform(x_features), y_features)
        final_train_models_gb.append(random_search)
    
    #Overall model
    x_features, y_features = extract_data(np.concatenate(x_train_list), np.concatenate(y_train_list), window)
    
    #Scaler to standerize data
    scaler = StandardScaler()
    scaler.fit(x_features)
    
    #Linear SVM
    random_search = RandomizedSearchCV(svm_linear, param_distributions=param_dist_linear, 
                                       refit='accuracy_score', n_iter=n_iter_search, 
                                       cv=5, verbose=10, scoring=scorers, n_jobs=-1)
    random_search.fit(scaler.transform(x_features), y_features)
    final_train_models_linear.append(random_search)
    
    #RBF SVM
    random_search = RandomizedSearchCV(svm_rbf, param_distributions=param_dist_rbf, 
                                       refit='accuracy_score', n_iter=n_iter_search, 
                                       cv=5, verbose=10, scoring=scorers, n_jobs=-1)
    random_search.fit(scaler.transform(x_features), y_features)
    final_train_models_rbf.append(random_search)
    
    #GradientBoosting
    random_search = RandomizedSearchCV(gb, param_distributions=param_dist_gb, 
                                       refit='accuracy_score', n_iter=n_iter_search, 
                                       cv=5, verbose=10, scoring=scorers, n_jobs=-1)
    random_search.fit(scaler.transform(x_features), y_features)
    final_train_models_gb.append(random_search)
    
    with open('final_linear.pkl', 'wb') as f:
        pickle.dump(final_train_models_linear, f)
        
    with open('final_rbf.pkl', 'wb') as f:
        pickle.dump(final_train_models_rbf, f)
        
    with open('final_gb.pkl', 'wb') as f:
        pickle.dump(final_train_models_gb, f)
    
    #Standerize values from train data
    scaler_list = []
    for x_i, y_i in zip(x_train_list, y_train_list):
        x_features, y_features = extract_data(x_i, y_i, window)
            
        #Scaler to standerize data
        scaler = StandardScaler()
        scaler.fit(x_features)
        scaler_list.append(scaler)
    
    x_features, y_features = extract_data(np.concatenate(x_train_list), np.concatenate(y_train_list), window)
    
    #Scaler to standerize data
    scaler = StandardScaler()
    scaler.fit(x_features)
    scaler_list.append(scaler)
        
    df_results = pd.DataFrame(columns=['model','type','pid','accuracy','recall','precision'])
    n_iter = 0
    for x_i, y_i in zip(x_test_list, y_test_list):
        individual_score_linear = getScores(final_train_models_linear[n_iter], scaler_list[n_iter].transform(x_i), y_i)
        overall_score_linear = getScores(final_train_models_linear[5], scaler_list[5].transform(x_i), y_i)
        df_results = append_score_to_df(df_results, 'linear', 'ind', n_iter, individual_score_linear.accuracy, individual_score_linear.recall, individual_score_linear.precision)
        df_results = append_score_to_df(df_results, 'linear', 'overall', n_iter, overall_score_linear.accuracy, overall_score_linear.recall, overall_score_linear.precision)
        
        individual_score_rbf = getScores(final_train_models_rbf[n_iter], scaler_list[n_iter].transform(x_i), y_i)
        overall_score_rbf = getScores(final_train_models_rbf[5], scaler_list[5].transform(x_i), y_i)
        df_results = append_score_to_df(df_results, 'rbf', 'ind', n_iter, individual_score_rbf.accuracy, individual_score_rbf.recall, individual_score_rbf.precision)
        df_results = append_score_to_df(df_results, 'rbf', 'overall', n_iter, overall_score_rbf.accuracy, overall_score_rbf.recall, overall_score_rbf.precision)
        
        individual_score_gb = getScores(final_train_models_gb[n_iter], scaler_list[n_iter].transform(x_i), y_i)
        overall_score_gb = getScores(final_train_models_gb[5], scaler_list[5].transform(x_i), y_i)
        df_results = append_score_to_df(df_results, 'gb', 'ind', n_iter, individual_score_gb.accuracy, individual_score_gb.recall, individual_score_gb.precision)
        df_results = append_score_to_df(df_results, 'gb', 'overall', n_iter, overall_score_gb.accuracy, overall_score_gb.recall, overall_score_gb.precision)
                
        n_iter += 1
    
    df_results.to_csv('final_results.csv', index=False)
    
#    df_win = pd.DataFrame(columns=['model','window','pid','mean_acc'])
#    
#    for win_i in [3,5,7,9,11,13,15,17,19]:
#        #Extract true blinks and non blinks for train data
#        x_train_list, y_train_list, x_test_list, y_test_list = getData('data', window=win_i)
#        n_iter = 0
#        for x_i,y_i in zip(x_train_list, y_train_list):
#            n_iter+=1
#            
#            x_features, y_features = extract_data(x_i, y_i, win_i)
#            
#            #Scaler to standerize data
#            scaler = StandardScaler()
#            scaler.fit(x_features)
#            
#            #Linear SVM
#            random_search = RandomizedSearchCV(svm_linear, param_distributions=param_dist_linear, 
#                                               refit='accuracy_score', n_iter=n_iter_search, 
#                                               cv=5, verbose=10, scoring=scorers, n_jobs=-1)
#            random_search.fit(scaler.transform(x_features), y_features)
#            df_win = append_to_df(df_win, random_search.best_score_, 'svm_linear', win_i, n_iter)
#
#            #RBF SVM
#            random_search = RandomizedSearchCV(svm_rbf, param_distributions=param_dist_rbf, 
#                                               refit='accuracy_score', n_iter=n_iter_search, 
#                                               cv=5, verbose=10, scoring=scorers, n_jobs=-1)
#            random_search.fit(scaler.transform(x_features), y_features)
#            df_win = append_to_df(df_win, random_search.best_score_, 'svm_rbf', win_i, n_iter)
#            
#            #GradientBoosting
#            random_search = RandomizedSearchCV(gb, param_distributions=param_dist_gb, 
#                                               refit='accuracy_score', n_iter=n_iter_search, 
#                                               cv=5, verbose=10, scoring=scorers, n_jobs=-1)
#            random_search.fit(scaler.transform(x_features), y_features)
#            df_win = append_to_df(df_win, random_search.best_score_, 'gb', win_i, n_iter)
#        
#        #Overall model
#        x_features, y_features = extract_data(np.concatenate(x_train_list), np.concatenate(y_train_list), win_i)
#        
#        #Linear SVM
#        random_search = RandomizedSearchCV(svm_linear, param_distributions=param_dist_linear, 
#                                           refit='accuracy_score', n_iter=n_iter_search, 
#                                           cv=5, verbose=10, scoring=scorers, n_jobs=-1)
#        random_search.fit(scaler.transform(x_features), y_features)
#        df_win = append_to_df(df_win, random_search.best_score_, 'svm_linear', win_i, 'full')
#
#        #RBF SVM
#        random_search = RandomizedSearchCV(svm_rbf, param_distributions=param_dist_rbf, 
#                                           refit='accuracy_score', n_iter=n_iter_search, 
#                                           cv=5, verbose=10, scoring=scorers, n_jobs=-1)
#        random_search.fit(scaler.transform(x_features), y_features)
#        df_win = append_to_df(df_win, random_search.best_score_, 'svm_rbf', win_i, 'full')
#        
#        #GradientBoosting
#        random_search = RandomizedSearchCV(gb, param_distributions=param_dist_gb, 
#                                           refit='accuracy_score', n_iter=n_iter_search, 
#                                           cv=5, verbose=10, scoring=scorers, n_jobs=-1)
#        random_search.fit(scaler.transform(x_features), y_features)
#        df_win = append_to_df(df_win, random_search.best_score_, 'gb', win_i, 'full')
#    df_win.to_csv('param_win.cvs', index=False)   

if __name__ == '__main__':
    main()