from flask import Flask, jsonify, request
from energyplus_api_helper import EnergyPlusAPIHelper

app = Flask(__name__)
api_helper = EnergyPlusAPIHelper()

@app.route('/vav_box', methods=['GET', 'POST'])
def vav_box():
    if request.method == 'GET':
        # Get the current VAV box settings
        vav_box_settings = api_helper.get_vav_box_settings()
        return jsonify(vav_box_settings)
    elif request.method == 'POST':
        # Set the VAV box settings
        data = request.get_json()
        api_helper.set_vav_box_settings(data)
        return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)