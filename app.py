from flask import Flask, request, Response,  redirect, render_template, url_for, session, flash
from werkzeug.utils import secure_filename
import ConfigParser
import json
import time
import os
import ipdb
#from train_dev import Net
app = Flask(__name__)
app.config.from_object(__name__)
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

# Load default config and override config from an environment variable
app.config.update(dict(
    MODEL_DIR = os.path.join(app.root_path, 'static', 'data')
))
app.config.from_envvar('FLASK_SETTINGS', silent=True)

def write_config(config_info, config_file):
    config = ConfigParser.ConfigParser()
    config.add_section('data')
    config.set('data', 'train_file', config_info['train_file'])
    config.set('data', 'dev_file', config_info['dev_file'])
    config.add_section('input')
    config.set('input', 'maxTokenLength', config_info['maxTokenLength'])
    config.set('input', 'embedding_size', config_info['embedding_size'])
    config.set('input', 'channel_num', len(config_info['channel']))
    for idx, channel in enumerate(config_info['channel']):
        channel_name = 'channel'+str(idx)
        config.add_section(channel_name)
        config.set(channel_name, 'mode', channel['mode'])
        config.set(channel_name, 'static', channel['static'])
        config.set(channel_name, 'embedding', channel['embedding'])
    config.add_section('net')
    config.set('net', 'filter_size', ','.join(config_info['filter_size']))
    config.set('net', 'filter_num', config_info['filter_num'])
    config.add_section('output')
    config.set('output', 'workspace',config_info['workspace'])
    with open(config_file, 'w+') as f:
        config.write(f)
    return config_file

    
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/config", methods = ['POST', 'GET'])
def get_config_data():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        config_info = {}
        config_info['embedding_size'] = request.form['embedding_size']
        config_info['maxTokenLength'] = request.form['maxTokenLength']
        mode = request.form.getlist('mode')
        static = request.form.getlist('static')
        embedding = request.form.getlist('embedding')
        channel = []
        for i in range(len(mode)):
            channel.append({'mode':mode[i], 'static':static[i], 'embedding':embedding[i]})
        config_info['channel'] = channel
        config_info['filter_size'] = request.form.getlist('filter_size')
        config_info['filter_num'] = '128'
        #model_dir = os.path.join(app.config['MODEL_DIR'], 'model')
        model_dir = app.config['MODEL_DIR']
        train_file = request.files['train_file']
        if train_file.filename == '':
            flash('No selected train file')
            return redirect(request.url)
        else:
            train_file_path = os.path.join(model_dir, secure_filename(train_file.filename))
            train_file.save(train_file_path)
            config_info['train_file'] = train_file_path
        dev_file = request.files['dev_file']
        if dev_file.filename == '':
            flash('No selected dev file')
            return redirect(request.url)
        else:
            dev_file_path = os.path.join(model_dir, secure_filename(dev_file.filename))
            dev_file.save(dev_file_path)
            config_info['dev_file'] = dev_file_path
        config_info['workspace'] = model_dir
        config_file = write_config(config_info, os.path.join(model_dir,'config.ini'))
        session['config_file'] = config_file
        session['config_info'] = config_info
        return redirect(url_for('train_model'))
        #return redirect(url_for('.train_model', config_file=config_file))

@app.route("/train", methods = ['GET', 'POST'])
def train_model():
    config_file = session['config_file']
    config_info = session['config_info']
    if request.method == 'GET':
        return render_template('train.html', config_file=config_file, config_info=config_info)
    else:
        epoch_num = int(request.form['epoch_num'])
        batch_size = int(request.form['batch_size'])

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5001,  threaded=True)
