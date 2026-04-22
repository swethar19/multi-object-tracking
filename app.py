from flask import Flask, render_template, redirect, url_for, request, session, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os, threading, time

app = Flask(__name__)
app.secret_key = 'trackvision_secret_2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///trackvision.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('uploads', 'videos')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

db = SQLAlchemy(app)

# ─── Models ───────────────────────────────────────────────
class User(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    username  = db.Column(db.String(80),  unique=True, nullable=False)
    email     = db.Column(db.String(120), unique=True, nullable=False)
    password  = db.Column(db.String(200), nullable=False)
    created   = db.Column(db.DateTime, default=db.func.now())

with app.app_context():
    db.create_all()

# ─── Camera / streaming state ─────────────────────────────
camera_state = {
    'person':  {'active': False, 'frame': None, 'lock': threading.Lock(), 'source': None},
    'vehicle': {'active': False, 'frame': None, 'lock': threading.Lock(), 'source': None},
}

# ─── Lazy-load tracking modules ───────────────────────────
_person_gen  = None
_vehicle_gen = None

def get_person_generator():
    global _person_gen
    if _person_gen is None:
        from tracker.person_tracker import PersonTracker
        _person_gen = PersonTracker()
    return _person_gen

def get_vehicle_generator():
    global _vehicle_gen
    if _vehicle_gen is None:
        from tracker.vehicle_tracker import VehicleTracker
        _vehicle_gen = VehicleTracker()
    return _vehicle_gen

# ─── Auth helpers ─────────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ─── Routes ───────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        u = request.form['username'].strip()
        e = request.form['email'].strip()
        p = request.form['password']
        if User.query.filter((User.username==u)|(User.email==e)).first():
            return render_template('register.html', error='Username or email already exists.')
        user = User(username=u, email=e, password=generate_password_hash(p))
        db.session.add(user); db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        u = request.form['username'].strip()
        p = request.form['password']
        user = User.query.filter_by(username=u).first()
        if user and check_password_hash(user.password, p):
            session['user_id']  = user.id
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        return render_template('login.html', error='Invalid credentials.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=session['username'])

@app.route('/person-track')
@login_required
def person_track():
    return render_template('person_track.html', username=session['username'])

@app.route('/vehicle-detect')
@login_required
def vehicle_detect():
    return render_template('vehicle_detect.html', username=session['username'])

# ─── Person tracking ──────────────────────────────────────
@app.route('/api/person/start-live', methods=['POST'])
@login_required
def person_start_live():
    state = camera_state['person']
    if state['active']:
        return jsonify({'status':'already running'})
    state['active'] = True
    state['source'] = 'live'
    def run():
        gen = get_person_generator()
        for frame_bytes in gen.generate_live():
            if not state['active']:
                break
            with state['lock']:
                state['frame'] = frame_bytes
        state['active'] = False
    threading.Thread(target=run, daemon=True).start()
    return jsonify({'status':'started'})

@app.route('/api/person/stop', methods=['POST'])
@login_required
def person_stop():
    camera_state['person']['active'] = False
    time.sleep(0.3)
    return jsonify({'status':'stopped'})

@app.route('/api/person/upload', methods=['POST'])
@login_required
def person_upload():
    state = camera_state['person']
    if 'video' not in request.files:
        return jsonify({'error':'No file'}), 400
    f = request.files['video']
    fname = secure_filename(f.filename)
    path  = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    f.save(path)
    if state['active']:
        state['active'] = False
        time.sleep(0.5)
    state['active'] = True
    state['source'] = 'video'
    def run():
        gen = get_person_generator()
        for frame_bytes in gen.generate_video(path):
            if not state['active']:
                break
            with state['lock']:
                state['frame'] = frame_bytes
        state['active'] = False
    threading.Thread(target=run, daemon=True).start()
    return jsonify({'status':'started'})

@app.route('/stream/person')
@login_required
def stream_person():
    def gen():
        while True:
            state = camera_state['person']
            if not state['active']:
                time.sleep(0.05)
                continue
            with state['lock']:
                frame = state['frame']
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ─── Vehicle detection ────────────────────────────────────
@app.route('/api/vehicle/start-live', methods=['POST'])
@login_required
def vehicle_start_live():
    state = camera_state['vehicle']
    if state['active']:
        return jsonify({'status':'already running'})
    state['active'] = True
    state['source'] = 'live'
    def run():
        gen = get_vehicle_generator()
        for frame_bytes in gen.generate_live():
            if not state['active']:
                break
            with state['lock']:
                state['frame'] = frame_bytes
        state['active'] = False
    threading.Thread(target=run, daemon=True).start()
    return jsonify({'status':'started'})

@app.route('/api/vehicle/stop', methods=['POST'])
@login_required
def vehicle_stop():
    camera_state['vehicle']['active'] = False
    time.sleep(0.3)
    return jsonify({'status':'stopped'})

@app.route('/api/vehicle/upload', methods=['POST'])
@login_required
def vehicle_upload():
    state = camera_state['vehicle']
    if 'video' not in request.files:
        return jsonify({'error':'No file'}), 400
    f = request.files['video']
    fname = secure_filename(f.filename)
    path  = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    f.save(path)
    if state['active']:
        state['active'] = False
        time.sleep(0.5)
    state['active'] = True
    state['source'] = 'video'
    def run():
        gen = get_vehicle_generator()
        for frame_bytes in gen.generate_video(path):
            if not state['active']:
                break
            with state['lock']:
                state['frame'] = frame_bytes
        state['active'] = False
    threading.Thread(target=run, daemon=True).start()
    return jsonify({'status':'started'})

@app.route('/stream/vehicle')
@login_required
def stream_vehicle():
    def gen():
        while True:
            state = camera_state['vehicle']
            if not state['active']:
                time.sleep(0.05)
                continue
            with state['lock']:
                frame = state['frame']
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)
