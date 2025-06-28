import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime
import time
from scipy.spatial.distance import euclidean
from dtaidistance import dtw 
import hashlib

class HandGestureAuth:
    def __init__(self):
        # Initialize Mediapipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Authentication parameters
        self.similarity_threshold = 0.15  # Lower = more strict
        self.min_gesture_frames = 20      # Minimum frames to capture
        self.max_gesture_frames = 60      # Maximum frames to capture
        
        # Storage
        self.users_db_path = "users_gestures.json"
        self.users_db = self.load_users_db()
        
        # Tracking variables
        self.current_gesture = []
        self.recording = False
        self.frame_count = 0
        
    def load_users_db(self):
        """Load users database from JSON file"""
        if os.path.exists(self.users_db_path):
            with open(self.users_db_path, 'r') as f:
                data = json.load(f)
                
            # Convert lists back to numpy arrays
            converted_db = {}
            for username, user_data in data.items():
                converted_db[username] = {
                    'gestures': [
                        [np.array(frame) if isinstance(frame, list) else frame 
                         for frame in gesture]
                        for gesture in user_data['gestures']
                    ],
                    'registered_at': user_data['registered_at'],
                    'user_id': user_data['user_id']
                }
            return converted_db
        return {}
    
    def save_users_db(self):
        """Save users database to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        serializable_db = {}
        for username, user_data in self.users_db.items():
            serializable_db[username] = {
                'gestures': [
                    [frame.tolist() if isinstance(frame, np.ndarray) else frame 
                     for frame in gesture]
                    for gesture in user_data['gestures']
                ],
                'registered_at': user_data['registered_at'],
                'user_id': user_data['user_id']
            }
        
        with open(self.users_db_path, 'w') as f:
            json.dump(serializable_db, f, indent=2)
    
    def extract_landmarks(self, frame):
        """Extract hand landmarks from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            
            # Extract x, y coordinates for all 21 landmarks
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y])
            
            return np.array(landmarks), hand_landmarks
        
        return None, None
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks to be translation and scale invariant"""
        if landmarks is None or len(landmarks) == 0:
            return None
            
        # Reshape to (21, 2) for easier processing
        points = landmarks.reshape(-1, 2)
        
        # Get wrist position (landmark 0) as reference
        wrist = points[0]
        
        # Translate to make wrist the origin
        centered = points - wrist
        
        # Scale normalization using distance between wrist and middle finger tip
        middle_tip = centered[12]  # Middle finger tip
        scale = np.linalg.norm(middle_tip)
        
        if scale > 0:
            normalized = centered / scale
        else:
            normalized = centered
            
        return normalized.flatten()
    
    def calculate_gesture_similarity(self, gesture1, gesture2):
        """Calculate similarity between two gestures using DTW or fallback method"""
        if len(gesture1) == 0 or len(gesture2) == 0:
            return float('inf')
        
        # Convert to numpy arrays
        g1 = np.array(gesture1)
        g2 = np.array(gesture2)
        
        # Try DTW first, fallback to simple method
        try:
            # Uncomment next line if dtaidistance is installed
            # from dtaidistance import dtw
            # distance = dtw.distance(g1, g2)
            # return distance
            pass
        except:
            pass
        
        # Fallback method: Average frame-wise euclidean distance
        # Resample to same length for comparison
        min_len = min(len(g1), len(g2))
        if min_len == 0:
            return float('inf')
            
        # Simple resampling
        indices1 = np.linspace(0, len(g1)-1, min_len).astype(int)
        indices2 = np.linspace(0, len(g2)-1, min_len).astype(int)
        
        resampled_g1 = g1[indices1]
        resampled_g2 = g2[indices2]
        
        # Calculate average euclidean distance
        distances = [euclidean(f1, f2) for f1, f2 in zip(resampled_g1, resampled_g2)]
        return np.mean(distances)
    
    def register_user(self, username):
        """Register a new user by capturing their gesture"""
        print(f"\n=== REGISTERING USER: {username} ===")
        print("Instructions:")
        print("1. Position your hand in front of the camera")
        print("2. Press 'r' to start recording your gesture")
        print("3. Perform your unique gesture slowly and clearly")
        print("4. Press 'r' again to stop recording")
        print("5. Repeat this process 3 times")
        print("6. Press 'q' to quit\n")
        
        cap = cv2.VideoCapture(0)
        user_gestures = []
        attempt = 1
        
        while len(user_gestures) < 3:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)  # Mirror the frame
            landmarks, hand_landmarks = self.extract_landmarks(frame)
            
            # Draw hand landmarks if detected
            if hand_landmarks is not None:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
            
            # Display recording status
            status_text = f"Attempt {attempt}/3 - "
            if self.recording:
                status_text += f"RECORDING... ({self.frame_count}/{self.max_gesture_frames})"
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Record gesture
                if landmarks is not None:
                    normalized = self.normalize_landmarks(landmarks)
                    if normalized is not None:
                        self.current_gesture.append(normalized)
                        self.frame_count += 1
                
                # Auto-stop if max frames reached
                if self.frame_count >= self.max_gesture_frames:
                    self.recording = False
                    if len(self.current_gesture) >= self.min_gesture_frames:
                        user_gestures.append(self.current_gesture.copy())
                        print(f"✓ Gesture {attempt} captured successfully! ({len(self.current_gesture)} frames)")
                        attempt += 1
                    else:
                        print(f"✗ Gesture too short. Please try again.")
                    
                    self.current_gesture = []
                    self.frame_count = 0
            else:
                status_text += "Press 'r' to start recording"
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show hand detection status
            detection_text = "Hand Detected: " + ("YES" if landmarks is not None else "NO")
            cv2.putText(frame, detection_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.imshow('Register User', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                if not self.recording:
                    print(f"Starting gesture recording {attempt}...")
                    self.recording = True
                    self.current_gesture = []
                    self.frame_count = 0
                else:
                    self.recording = False
                    if len(self.current_gesture) >= self.min_gesture_frames:
                        user_gestures.append(self.current_gesture.copy())
                        print(f"✓ Gesture {attempt} captured successfully! ({len(self.current_gesture)} frames)")
                        attempt += 1
                    else:
                        print(f"✗ Gesture too short ({len(self.current_gesture)} frames). Minimum {self.min_gesture_frames} required.")
                    
                    self.current_gesture = []
                    self.frame_count = 0
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(user_gestures) == 3:
            # Store user gestures
            self.users_db[username] = {
                'gestures': user_gestures,
                'registered_at': datetime.now().isoformat(),
                'user_id': hashlib.md5(username.encode()).hexdigest()[:8]
            }
            self.save_users_db()
            print(f"✓ User '{username}' registered successfully!")
            return True
        else:
            print(f"✗ Registration failed. Only captured {len(user_gestures)}/3 gestures.")
            return False
    
    def authenticate_user(self, username):
        """Authenticate user by comparing their gesture"""
        if username not in self.users_db:
            print(f"User '{username}' not found!")
            return False
        
        print(f"\n=== AUTHENTICATING USER: {username} ===")
        print("Instructions:")
        print("1. Position your hand in front of the camera")
        print("2. Press 'a' to start authentication")
        print("3. Perform your registered gesture")
        print("4. Press 'q' to quit\n")
        
        cap = cv2.VideoCapture(0)
        authenticated = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            landmarks, hand_landmarks = self.extract_landmarks(frame)
            
            if hand_landmarks is not None:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
            
            # Display status
            if self.recording:
                status_text = f"AUTHENTICATING... ({self.frame_count}/{self.max_gesture_frames})"
                cv2.putText(frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Record gesture
                if landmarks is not None:
                    normalized = self.normalize_landmarks(landmarks)
                    if normalized is not None:
                        self.current_gesture.append(normalized)
                        self.frame_count += 1
                
                # Auto-stop and authenticate
                if self.frame_count >= self.max_gesture_frames or len(self.current_gesture) >= self.min_gesture_frames:
                    self.recording = False
                    
                    if len(self.current_gesture) >= self.min_gesture_frames:
                        # Compare with stored gestures
                        user_gestures = self.users_db[username]['gestures']
                        min_distance = float('inf')
                        
                        for stored_gesture in user_gestures:
                            distance = self.calculate_gesture_similarity(
                                self.current_gesture, stored_gesture
                            )
                            min_distance = min(min_distance, distance)
                        
                        print(f"Similarity distance: {min_distance:.4f}")
                        print(f"Threshold: {self.similarity_threshold:.4f}")
                        
                        if min_distance <= self.similarity_threshold:
                            print("✓ AUTHENTICATION SUCCESSFUL!")
                            authenticated = True
                            break
                        else:
                            print("✗ AUTHENTICATION FAILED!")
                    
                    self.current_gesture = []
                    self.frame_count = 0
            else:
                cv2.putText(frame, "Press 'a' to authenticate", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show hand detection status
            detection_text = "Hand Detected: " + ("YES" if landmarks is not None else "NO")
            cv2.putText(frame, detection_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.imshow('Authenticate User', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a') and not self.recording:
                print("Starting authentication...")
                self.recording = True
                self.current_gesture = []
                self.frame_count = 0
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return authenticated
    
    def list_users(self):
        """List all registered users"""
        if not self.users_db:
            print("No users registered yet.")
            return
        
        print("\n=== REGISTERED USERS ===")
        for username, data in self.users_db.items():
            registered_at = data.get('registered_at', 'Unknown')
            user_id = data.get('user_id', 'Unknown')
            gesture_count = len(data.get('gestures', []))
            print(f"• {username} (ID: {user_id}) - {gesture_count} gestures - {registered_at}")
    
    def delete_user(self, username):
        """Delete a user from the database"""
        if username in self.users_db:
            del self.users_db[username]
            self.save_users_db()
            print(f"✓ User '{username}' deleted successfully!")
            return True
        else:
            print(f"✗ User '{username}' not found!")
            return False

def main():
    """Main application loop"""
    auth_system = HandGestureAuth()
    
    print("🔐 Hand Gesture Authentication System - Phase 1 MVP")
    print("=" * 55)
    
    while True:
        print("\nSelect an option:")
        print("1. Register new user")
        print("2. Authenticate user")
        print("3. List all users")
        print("4. Delete user")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            username = input("Enter username: ").strip()
            if username:
                if username in auth_system.users_db:
                    overwrite = input(f"User '{username}' already exists. Overwrite? (y/n): ")
                    if overwrite.lower() != 'y':
                        continue
                auth_system.register_user(username)
            else:
                print("Username cannot be empty!")
                
        elif choice == '2':
            username = input("Enter username to authenticate: ").strip()
            if username:
                start_time = time.time()
                success = auth_system.authenticate_user(username)
                end_time = time.time()
                
                print(f"Authentication time: {end_time - start_time:.2f} seconds")
                if success:
                    print("🎉 ACCESS GRANTED!")
                else:
                    print("🚫 ACCESS DENIED!")
            else:
                print("Username cannot be empty!")
                
        elif choice == '3':
            auth_system.list_users()
            
        elif choice == '4':
            username = input("Enter username to delete: ").strip()
            if username:
                confirm = input(f"Are you sure you want to delete '{username}'? (y/n): ")
                if confirm.lower() == 'y':
                    auth_system.delete_user(username)
            else:
                print("Username cannot be empty!")
                
        elif choice == '5':
            print("Goodbye! 👋")
            break
            
        else:
            print("Invalid choice! Please enter 1-5.")

if __name__ == "__main__":
    main()