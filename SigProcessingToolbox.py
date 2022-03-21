#!/usr/bin/env python
from numpy import pi, sin, cos, arctan2, hypot, degrees, radians, dot, log10, arange, linspace, array
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import matplotlib.pyplot as plt
from itertools import repeat


SPEED_OF_SOUND = 343

@dataclass
class SimpleWave:
    frequency: float #Probably need to support more complicated waves.
    time: float
    
    @property
    def real_wave_part(self) -> float:
        """
        Returns the real part of a wave.
        """
        return cos(2.0*pi*self.frequency*self.time)
    
    @property
    def imaginary_wave_part(self) -> float:
        """
        Returns the imaginary part of a wave.
        """
        return sin(2.0*pi*self.frequency*self.time)
     
    @property
    def wave(self) -> set:
        """
        Returns a set of the real and imaginary parts of a wave
        """
        return (self.real_wave_part, self.imaginary_wave_part)
    
    @property
    def phase(self) -> float:
        # Phase is equal to the 180*arctan(imaginary/real)/pi
        return 180*arctan2(self.imaginary_wave_part, self.real_wave_part)/pi
    
    @property
    def amplitude(self) -> float:
        # Amplitude is equal to the Euclidean norm of real part of the wave plus the impaginar portion of the wave
        return hypot(self.real_wave_part, self.imaginary_wave_part)
        
    def __add__(self, SimpleWave):
        """
        Should take a SimpleWave, add it the current class instance, and return a class that is the sum of the two. 
        Maybe should return a ComplexWave class that better represents the waves added together. 
        I'm kinda feeling lazy though so this is more a speculating on what could be done.
        """
        #Hard
        pass
        
    
class Toolbox(object):
    @classmethod
    def farfield_mic_delay(cls, mic_offset: list, source: tuple) -> float:
        """
        Returns the delay of a farfield planar wavefront to reach an element from some 3D offset given a elevation and 
        azimuth for the signal.
        Input:
          mic_pos - set of [m_x, m_y, m_z] where m is the position of a mic in relation to a delay reference point.
          source - tuple of (elevation, azimuth)
        
        Returns:
          delay of signal between an element from some reference offset.
        """
        # convert source into a unit vector where source[0] is elevation source[1] is azimuth. Simplifies the math.
        source_x = cos(source[0]) * cos(source[1])
        source_y = cos(source[0]) * sin(source[1])
        source_z = sin(source[0])
        delay = dot(mic_offset, [source_x, source_y, source_z]) / SPEED_OF_SOUND
        return delay

    @classmethod
    def wave_sum(cls, *args) -> float:
        """
        Returns the sum of n number waves in their real and imaginary parts
        Input:
          takes n number of sets of wave
        
        Returns:
          sum of waves in their real and imaginary parts
        """
        # I bet I could overwrite the dunder method for __add__ in my SimpleWave class so that I could just add waves together 
        # as easy as wave3 = wave1 + wave2. That said, I think this may be more computationally efficient?
        return set(map(sum, zip(*args)))


def freq_linear_sensitivity(frequency, element_num, spacing, angle_resolution):
    # Basic delay-sum beamformer calculaticing sensitivity of the array for a single frequency. 
    rads = []
    decibals = []
    
    for a in range(angle_resolution):
        rad = radians(180 * a / (angle_resolution - 1) - 90)
        # farfield_mic_delay assumes a plane wave. I wrote it to support three dimensional beamforming but works
        # just fine for linear beamforming.
        delay = lambda elem: Toolbox.farfield_mic_delay((0, elem * spacing, 0), (0, rad))
        linear_array_waves_at_angle = [SimpleWave(frequency, delay(element)).wave for element in range(element_num)]
        
        # This code should honestly be part of complexWave class. Probably overwrite the __add__ dunder method 
        # Sums of the real and imaginary number wave elements.
        wave_sum = Toolbox.wave_sum(*linear_array_waves_at_angle)
        # Determine amplitude
        output = hypot(*wave_sum) / element_num
        
        # Convert to logarithmic decibel scale
        decibal = 20 * log10(output)
        if decibal < -50:
            decibal = -50
        # Can't pickle generators but, otherwise, my preference is to yield the result so I don't have to hold that
        # Data in memory as a full array. My first example can easily consume a generator.
        #yield rad, decibal 
        rads.append(rad)
        decibals.append(decibal)
    return rads, decibals
    

def frequence_range_linear_sensitivity(frequency_start, frequency_end, frequence_resolution, element_num, spacing, angle_resolution):
    """ 
        Delay-sum beamformer calculaticing sensitivity of the array for a frequency spectrum.
    """
    # A list composed of frquencies from frequency_start to frequency_end determined by frequency resolution.
    freq_range = lambda f_range: [(frequency_end - frequency_start) * f / (frequence_resolution-1) + frequency_start for f in range(f_range)]
    gain_data = []
    with ProcessPoolExecutor() as exe:
        '''
            I think it's ridiculous that pickle can't handle lambda functions. I should just be able to do
            exe.map(lambda f: freq_linear_sensitivity(f, **kwargs), freq_range(frequence_resolution))
            Also, spinning up a subprocess for every frequency probably is a bad practice.
            I should have a subprocess handle a band of frequencies.
        '''
        for _, db_sensitivities in exe.map(freq_linear_sensitivity, freq_range(frequence_resolution), repeat(element_num), repeat(spacing), repeat(angle_resolution)):
            gain_data.append(db_sensitivities)
    # I wanted to be able to have frequency and gain data be yielded so I didn't have to store in memory but couldn't find a way to have matlibplot consume a generator.
    freq_data = [list(repeat(f, angle_resolution)) for f in freq_range(frequence_resolution)]
    return freq_data, gain_data
                
       
rads, decibals = freq_linear_sensitivity(frequency = 1000,
    element_num = 10,
    spacing = 0.2,
    angle_resolution = 500)

plt.axes(projection = 'polar')
# plotting the circle
for rad, decibal in zip(rads, decibals):
    plt.polar(rad, decibal, 'g.')
plt.title('single_freq_linear_sensitivity example\n', 
          fontsize = 9, fontweight ='bold')
plt.show()

freq_data, gain_data = frequence_range_linear_sensitivity(frequency_start = 0, frequency_end = 1000,
    element_num = 10,
    spacing = 0.2,
    frequence_resolution = 200,
    angle_resolution = 200)
X = linspace(-90,90, 200)
Y = array(freq_data)
Z = array(gain_data)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')

ax.set_xlabel('Arrival Angle (Degrees)')
ax.set_ylabel('Frequency (Hertz)')
ax.set_zlabel('Gain (Db)');
ax.set_title('frequence_range_linear_sensitivity example') 
plt.show()
