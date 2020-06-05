import React, {useEffect, useState} from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Image, ActivityIndicator, StatusBar, Button } from 'react-native';
import * as TF from '@tensorflow/tfjs';
import * as MobileNet from '@tensorflow-models/mobilenet';
import {fetch} from '@tensorflow/tfjs-react-native';
import Constants from 'expo-constants';
import * as ImagePicker from 'expo-image-picker';
import * as Permissions from 'expo-permissions';
import * as FileSystem from 'expo-file-system';
import * as jpeg from 'jpeg-js';

import {stringToUint8Array} from './helper';
import aImage from './assets/images.jpeg';

export default function App() {
  const [isTfReady, setIsTfReady] = useState(false);
  const [isModelReady, setIsModelReady] = useState(false);
  const [isCustomModelReady, setIsCustomModelReady] = useState(false);
  const [model, setModel] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [image, setImage] = useState(aImage);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function check() {
      await TF.ready();
      setIsTfReady(true);
      try {
        setModel(await MobileNet.load());
        setIsModelReady(true);
        console.log('.....')
        // console.log('NAT_MODEL', NAT_MODEL);
        const URL = 'https://tfhub.dev/google/tfjs-model/inaturalist/inception_v3/feature_vector/1/default/1';
        const mdl = await TF.loadGraphModel(URL, {fromTFHub: true});
        // const mdl = await TF.loadGraphModel(NAT_MODEL);
        console.log('model', mdl);
        setModel(mdl);
        setIsCustomModelReady(true);
      } catch(err) {
        console.log('err:::', err);
      }
    }
    const getPermissionAsync = async () => {
      if (Constants.platform.ios) {
        const { status } = await Permissions.askAsync(Permissions.CAMERA_ROLL)
        if (status !== 'granted') {
          alert('Sorry, we need camera roll permissions to make this work!')
        }
      }
    }
    check();
    getPermissionAsync();
  }, [])

  return (
    <View style={styles.container}>
        <StatusBar barStyle='light-content' />
        <View style={styles.loadingContainer}>
          <Text style={styles.text}>
            TFJS ready? {isTfReady ? '✅' : ''}
          </Text>

          <View style={styles.loadingModelContainer}>
            <Text style={styles.text}>Model ready? </Text>
            {isModelReady ? (
              <Text style={styles.text}>✅</Text>
            ) : (
              <ActivityIndicator size='small' />
            )}
            {isCustomModelReady ? (
              <Text style={styles.text}>✅</Text>
            ) : (
              <ActivityIndicator size='small' />
            )}
          </View>
        </View>
        <TouchableOpacity
          style={styles.imageWrapper}
          onPress={isModelReady ? selectImage : undefined}>
          {image && <Image source={image} style={styles.imageContainer} />}

          {isModelReady && !image && (
            <Text style={styles.transparentText}>Tap to choose image</Text>
          )}
        </TouchableOpacity>
        <Button onPress={classifyImage} title="Identify!" color="#841584" disabled={!isModelReady} />
        <View style={styles.predictionWrapper}>
          {isModelReady &&
            predictions &&
          predictions.map(({className, probability}) => <Text style={styles.text} key={className}>{className}: {probability}</Text>)}
        </View>
        {error && <Text style={[styles.text, {color: 'red'}]}>{error}</Text>}
      </View>
  );

  function imageToTensor(rawImageData) {
    const TO_UINT8ARRAY = true;
    const { width, height, data } = jpeg.decode(rawImageData, TO_UINT8ARRAY);
    // Drop the alpha channel info for MobileNet
    const buffer = new Uint8Array(width * height * 3)
    let offset = 0 // offset into original data
    for (let i = 0; i < buffer.length; i += 3) {
      buffer[i] = data[offset];
      buffer[i + 1] = data[offset + 1];
      buffer[i + 2] = data[offset + 2];

      offset += 4;
    }

    return TF.tensor3d(buffer, [height, width, 3]);
  }

  async function classifyImage() {
    try {
      const {uri} = Image.resolveAssetSource(image);
      let rawImageData;
      if(uri.startsWith('http') || uri.startsWith('https')) {
        const response = await fetch(uri, {}, { isBinary: true });
        rawImageData = await response.arrayBuffer();
      } else {
        const base64Img = await FileSystem.readAsStringAsync(uri, {encoding: FileSystem.EncodingType.Base64});
        rawImageData = stringToUint8Array(base64Img);
      }
      const imageTensor = imageToTensor(rawImageData);
      const predictions = await model.classify(imageTensor);
      setPredictions(predictions);
    } catch (error) {
      console.log(error);
      setError('CLASSIFY ERROR: ' + error.message);
    }
  };

  async function selectImage() {
    try {
      let response = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3]
      });
      if (!response.cancelled) {
        const source = { uri: response.uri };
        setImage(source);
      }
    } catch (error) {
      console.log(error);
      setError('SELECT IMG ERROR: ' + error.message);
    }
  }
}


const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#171f24',
    alignItems: 'center'
  },
  loadingContainer: {
    marginTop: 80,
    justifyContent: 'center'
  },
  text: {
    color: '#ffffff',
    fontSize: 16
  },
  loadingModelContainer: {
    flexDirection: 'row',
    marginTop: 10
  },
  imageWrapper: {
    width: 280,
    height: 280,
    padding: 10,
    borderColor: '#cf667f',
    borderWidth: 5,
    borderStyle: 'dashed',
    marginTop: 40,
    marginBottom: 10,
    position: 'relative',
    justifyContent: 'center',
    alignItems: 'center'
  },
  imageContainer: {
    width: 250,
    height: 250,
    position: 'absolute',
    top: 10,
    left: 10,
    bottom: 10,
    right: 10
  },
  predictionWrapper: {
    height: 100,
    width: '100%',
    flexDirection: 'column',
    alignItems: 'center'
  },
  transparentText: {
    color: '#ffffff',
    opacity: 0.7
  },
  footer: {
    marginTop: 40
  },
  tfLogo: {
    width: 125,
    height: 70
  }
});
