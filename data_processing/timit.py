import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import h5py

class TIMIT(Dataset):
    _ext_txt = ".TXT"
    _ext_phone = ".PHN"
    _ext_word = ".WRD"
    _ext_audio = ".WAV"
    _BLANK_LABEL = '_B_'
    '''
    mapping from:
    Lee, K. and Hon, H. Speaker-independent phone recognition using hidden markov models. 
    IEEE Transactions on Acoustics, Speech, and Signal Processing, 1989.
    '''
    _phone_map_48 ={
        'ux':'uw',
        'axr':'er',
        'em':'m',
        'nx':'n',
        'eng':'ng',
        'hv':'hh',
        'pcl': 'cl',
        'tcl': 'cl',
        'kcl': 'cl',
        'qcl': 'cl',
        'bcl': 'vcl',
        'gcl': 'vcl',
        'dcl': 'vcl',
        'h#':'sil',
        '#h':'sil',
        'pau':'sil',
        'ax-h':'ax',
        "q": None
    }
    _phone_map_39 = {
        'cl':'sil',
        'vcl': 'sil',
        'epi': 'sil',
        'el':'l',
        'en':'n',
        'sh':'zh',
        'ao':'aa',
        'ih':'ix',
        'ah':'ax'
    }
    def __getitem__(self, n):
        return self.load_timit_item(n)

    def __len__(self):
        return len(self._fileid_list)

    def __init__(self, phase, output_dir, data_dir, is_cache):
        self.output_dir = output_dir
        self.audio_h5_cache_filename = phase + '_timit_mfcc.h5'
        self.fileid_list_filename = phase + '_fileids.list'
        if is_cache:
            self.cache_audio_data(data_dir)
        if phase == 'train' and is_cache:
            self.save_stats()
            self.dump_phone_vocab()
        self.init_dataset()

    def load_stat_vocab(self, output_dir):
        self.read_stats(output_dir)
        self.read_vocab(output_dir)

    def get_file_ids(self, data_dir):
        walker = []
        for curr_root, _, fnames in sorted(os.walk(data_dir)):
            for fname in fnames:
                if fname.endswith(self._ext_phone):
                    walker.append(os.path.join(curr_root, fname[:-len(self._ext_phone)]))
        return list(walker)

    def init_dataset(self):
        h5_db = h5py.File(os.path.join(self.output_dir, self.audio_h5_cache_filename), 'r')
        self.mfcc_numpy_array = h5_db['mfcc']
        self.mfcc_lengths = h5_db['mfcc_lengths']
        fileid_list = []
        for fileid in open(os.path.join(self.output_dir, self.fileid_list_filename), 'r'):
            fileid_list.append(fileid.replace('\n', ''))
        self._fileid_list = fileid_list

    def get_audio_features(self, fileid):
        '''
        Standard speech preprocessing was applied to transform the audio files into feature sequences. 26 channel
        mel-frequency filter bank and a pre-emphasis coefficient of 0.97 were used to compute 12 mel-frequency
        cepstral coefficients plus an energy coefficient on 25ms Hamming windows at 10ms intervals.
        Delta coefficients were added to create input sequences of length 26 vectors, and all coefficient
        were normalised to have mean zero and standard deviation one over the training set.
        '''
        file_audio = fileid + self._ext_audio
        d, sr = librosa.load(file_audio, sr=None)
        d = librosa.effects.preemphasis(d)
        hop_length = int(0.010 * sr)
        n_fft = int(0.025 * sr)
        mfcc = librosa.feature.mfcc(d, sr, n_mfcc=13,
                                     hop_length=hop_length,
                                     n_fft=n_fft, window='hamming')
        mfcc[0] = librosa.feature.rms(y=d,
                                       hop_length=hop_length,
                                       frame_length=n_fft)
        deltas1 = librosa.feature.delta(mfcc, order=1)
        deltas2 = librosa.feature.delta(mfcc, order=2)
        mfccs_plus_deltas = np.vstack([mfcc, deltas1, deltas2])

        #individual wave file normalization - power norm might does this so not needed
        #mfccs_plus_deltas -= (np.mean(mfccs_plus_deltas, axis=0) + 1e-8)
        mfccs_plus_deltas = mfccs_plus_deltas.swapaxes(0, 1)
        return mfccs_plus_deltas

    def dump_phone_vocab(self):
        fileid_list = []
        for fileid in open(os.path.join(self.output_dir, self.fileid_list_filename), 'r'):
            fileid_list.append(fileid.replace('\n', ''))
        phone_vocab = []
        for fileid in fileid_list:
            audio_phones = self.load_phone_item(fileid)
            for audio_phone in audio_phones:
                if audio_phone['phone'] not in phone_vocab:
                    phone_vocab.append(audio_phone['phone'])
        with open(os.path.join(self.output_dir, 'phone.vocab'), 'w', encoding='utf8') as f:
            f.write('\n'.join(sorted(phone_vocab)))

    def cache_audio_data(self, data_dir):
        all_mfccs = []
        lengths = []
        fileid_list = self.get_file_ids(data_dir)
        for fileid in fileid_list:
            mfcc = self.get_audio_features(fileid)
            length = len(mfcc)
            all_mfccs.append(mfcc)
            lengths.append(length)

        max_len = max(lengths)
        mfcc_numpy_array = np.zeros((len(lengths), max_len, all_mfccs[0].shape[1]))
        for i, length in enumerate(lengths):
            mfcc_numpy_array[i, :length] = all_mfccs[i][:length]

        with h5py.File(os.path.join(self.output_dir, self.audio_h5_cache_filename), 'w') as f:
            f.create_dataset('mfcc', data=mfcc_numpy_array)
            f.create_dataset('mfcc_lengths', data=np.array(lengths))

        with open(os.path.join(self.output_dir, self.fileid_list_filename), 'w') as f:
            f.write('\n'.join(fileid_list))

    def save_stats(self):
        #calculate mean / var
        with h5py.File(os.path.join(self.output_dir, self.audio_h5_cache_filename), 'r') as h5_db:
            mfcc_numpy_array = h5_db['mfcc']
            mfcc_lengths = h5_db['mfcc_lengths']
            flatten_mfcc = []
            for i in range(len(mfcc_lengths)):
                length = mfcc_lengths[i]
                mfcc = mfcc_numpy_array[i][:length]
                mfcc = np.vsplit(mfcc, length)
                flatten_mfcc.extend(mfcc)
            flatten_mfcc = np.stack(flatten_mfcc, axis=0).squeeze(axis=1)
            mean = np.expand_dims(flatten_mfcc.mean(axis=0), axis=0)
            var = np.expand_dims(flatten_mfcc.var(axis=0), axis=0)
            np.savez(os.path.join(self.output_dir, 'stats.npz'), mean=mean, var=var)

    def read_stats(self, output_dir):
        train_stats = np.load(os.path.join(output_dir, 'stats.npz'))
        self._mean = train_stats['mean']
        self._var = train_stats['var']

    def normalize(self, mfcc):
        return (mfcc - self._mean) / self._var

    def read_vocab(self, output_dir):
        phone_vocab = []
        for w in open(os.path.join(output_dir, 'phone.vocab'), 'r', encoding='utf8'):
            assert len(w) > 0
            phone_vocab.append(w.replace('\n', ''))

        self._phone_vocab = phone_vocab
        self._phone_vocab2id = {p:i for i, p in enumerate(phone_vocab)}

    def load_txt_item(self, fileid):
        file_text = fileid + self._ext_txt
        with open(file_text) as ft:
            audio_text = ft.readline().strip().lower().split()
        return {
            'start': int(audio_text[0]),
            'end'  : int(audio_text[1]),
            'words': audio_text[2:]
        }

    def load_word_item(self, fileid):
        file_text = fileid + self._ext_word
        audio_words = []
        with open(file_text) as ft:
            for line in ft:
                parts = line.strip().lower().split()
                audio_word = {
                    'start': int(parts[0]),
                    'end'  : int(parts[1]),
                    'word' : parts[2]
                }
                audio_words.append(audio_word)

        return audio_words

    def load_phone_item(self, fileid):
        file_phone = fileid + self._ext_phone
        audio_phones = []
        with open(file_phone) as fp:
            for line in fp:
                parts = line.strip().split()
                phone = parts[2]
                if phone in self._phone_map_48:
                    phone = self._phone_map_48[phone]
                if phone is None:
                    continue
                if phone in self._phone_map_39:
                    phone = self._phone_map_39[phone]
                audio_phone = {
                    'start': int(parts[0]),
                    'end': int(parts[1]),
                    'phone': phone
                }
                audio_phones.append(audio_phone)
        return audio_phones

    def load_timit_item(self, i):
        fileid = self._fileid_list[i]

        length = self.mfcc_lengths[i]
        mfcc = self.mfcc_numpy_array[i][:length]

        #mfcc1 = self.get_audio_features(fileid)
        #length1 = len(mfcc1)
        #assert np.allclose(mfcc, mfcc1) and length == length1

        mfcc = self.normalize(mfcc)

        audio_phones = self.load_phone_item(fileid)
        phones = [self._phone_vocab2id[audio_phone['phone']] for audio_phone in audio_phones]

        return {
            'mfcc':torch.from_numpy(mfcc).float(),
            'length': torch.LongTensor([length]),
            'phone':torch.LongTensor(phones)
        }

def variable_collate_fn(batch):
    pad_token_id = -1
    mfccs = []
    phones = []
    lengths = []
    for sample in batch:
        mfccs.append(sample['mfcc'])
        lengths.append(sample['length'])
        phones.append(sample['phone'])

    mfccs = pad_sequence(mfccs, batch_first=True, padding_value=pad_token_id)
    lengths = torch.cat(lengths)
    phones = pad_sequence(phones, batch_first=True, padding_value=pad_token_id)

    return {
        'mfcc': mfccs,
        'length': lengths,
        'phone': phones
    }
def get_dataloader(timit_dataset, batch_size, is_shuffle, num_worker = 0):
    dataloader = DataLoader(timit_dataset, batch_size=batch_size, shuffle=is_shuffle,
                            num_workers=num_worker, collate_fn=variable_collate_fn)
    return dataloader
if __name__ == '__main__':

    train_output_dir = os.path.join(os.path.expanduser('~'),
                              'neural_sequence_transduction/TIMIT/TRAIN')
    timit_dataset = TIMIT('train', train_output_dir, train_output_dir, False)
    timit_dataset.load_stat_vocab(train_output_dir)
    ''' 
    output_dir = os.path.join(os.path.expanduser('~'),
                              'neural_sequence_transduction/TIMIT/TEST')
    timit_dataset = TIMIT('test', output_dir, output_dir, False)
    timit_dataset.load_stat_vocab(train_output_dir)
    '''
    dataloader = get_dataloader(timit_dataset, 4, True)
    for i_batch, sample_batched in enumerate(dataloader):
        print(sample_batched['mfcc'].size())



