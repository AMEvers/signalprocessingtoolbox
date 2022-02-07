from numpy import pi, sin, cos, arctan2, hypot, degrees, radians, dot, log10, arange, linspace, array
import matplotlib.pyplot as plt
from itertoos import repeat
from dataclasses import dataclass

SPEED_OF_SOUND = 343


@dataclass
class ComplexWave:
    frequency: float
    time: float

    @property
    def real_wave_part(self) -> float:
        """
        Returns the real part of a wave.
        """
        return cos(2.0 * pi * self.frequency * self.time)

    @property
    def imaginary_wave_part(self) -> float:
        """
        Returns the imaginary part of a wave.
        """
        return sin(2.0 * pi * self.frequency * self.time)

    @property
    def wave(self) -> tuple:
        return self.real_wave_part, self.imaginary_wave_part

    @property
    def phase(self) -> float:
        # Phase is equal to the 180*arctan(imaginary/real)/pi
        return 180 * arctan2(self.imaginary_wave_part, self.real_wave_part) / pi

    @property
    def amplitude(self) -> float:
        # Amplitude is equal to the Euclidean norm of real part of the wave plus the impaginar portion of the wave
        return hypot(self.real_wave_part, self.imaginary_wave_part)


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
        # Delay = mic_offset.source/speed of sound
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
        # I bet I could overwrite the dunder method for __add__ in my ComplexWave class so that I could just add waves together 
        # as easy as wave3 = wave1 + wave2. That said, I think this may be more computationally efficient?
        return set(map(sum, zip(*args)))


def single_freq_linear_sensitivity(frequency, element_num, spacing, angle_resolution):
    # Basic delay-sum beamformer calculaticing sensitivity of the array for a single frequency. 
    for a in range(angle_resolution):
        rad = radians(180 * a / (angle_resolution - 1) - 90)
        # farfield_mic_delay assumes a plane wave. I'm wrote it to support three dimensional beamforming but works
        # just fine for linear beamforming.
        delay = lambda elem: Toolbox.farfield_mic_delay((0, elem * spacing, 0), (0, rad))
        # I'm being lazy here. This isn't scalable to hold in memory for thousands of elements. I should use a
        # generator.
        linear_array_waves_at_angle = [ComplexWave(frequency, delay(element)).wave for element in range(element_num)]
        # Sums of the real and imaginary number wave elements.
        wave_sum = Toolbox.wave_sum(*linear_array_waves_at_angle)
        output = hypot(*wave_sum) / element_num
        # Convert to logarithmic decibel scale
        log_of_output = 20 * log10(output)
        if log_of_output < -50:
            log_of_output = -50
        yield rad, log_of_output


def frequence_range_linear_sensitivity(frequency_start, frequency_end, element_num, spacing, frequence_resolution,
                                       angle_resolution):
    # Delay-sum beamformer calculaticing sensitivity of the array for a frequency spectrum.
    freq = lambda f: (frequency_end - frequency_start) * f / (frequence_resolution - 1) + frequency_start
    freq_gain = lambda f: [log_of_output for _, log_of_output in single_freq_linear_sensitivity(freq(f), element_num, spacing, angle_resolution)]
    gain_data = [freq_gain(f) for f in range(frequence_resolution)]
    freq_data = [list(repeat(freq(f), angle_resolution)) for f in range(frequence_resolution)]
    return freq_data, gain_data


linear_beamform = single_freq_linear_sensitivity(frequency=1000,
                                                 element_num=10,
                                                 spacing=0.2,
                                                 angle_resolution=500)

plt.axes(projection='polar')
# creating an array containing the radian values
rads = arange(-50, 0, 0.01)
# plotting the circle
for radian, r in linear_beamform:
    plt.polar(radian, r, 'g.')
plt.title('single_freq_linear_sensitivity example\n',
          fontsize=9, fontweight='bold')
plt.show()

linear_beamform = frequence_range_linear_sensitivity(frequency_start=0, frequency_end=1000,
                                                     element_num=10,
                                                     spacing=0.2,
                                                     frequence_resolution=200,
                                                     angle_resolution=200)
X = linspace(-90, 90, 200)
Y = array(linear_beamform[0])
Z = array(linear_beamform[1])
print(len(Z))
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('Arrival Angle (Degrees)')
ax.set_ylabel('Frequency (Hertz)')
ax.set_zlabel('Gain (Db)');
ax.set_title('frequence_range_linear_sensitivity example')
plt.show()
