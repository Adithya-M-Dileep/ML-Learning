import React from 'react';
import React, { useState } from 'react';
import { Calendar, momentLocalizer } from 'react-big-calendar';
import moment from 'moment';
import 'react-big-calendar/lib/css/react-big-calendar.css';

const localizer = momentLocalizer(moment);

class MyCalendar extends React.Component {
  state = {
    events: [],
    isScheduling: false,
    currentEvent: {
      start: null,
      end: null,
      title: '',
    },
  };
  [selectedType, setSelectedType] = useState('');
  [availableVehicles, setAvailableVehicles] = useState([]);

  // Sample data for available vehicles based on type
  vehicleOptions = {
    car: ['Maruthi 800', 'Maruthi Waganar', 'Indigo XL'],
    bike: ['Honda Activa', 'Suzuki Access', 'Yamaha FZ'],
  };

   handleTypeChange = (e) => {
    const type = e.target.value;
    setSelectedType(type);
    setAvailableVehicles(vehicleOptions[type] || []);
  };

  handleSelectSlot = ({ start, end }) => {
    if (!this.state.isScheduling) {
      this.setState({
        isScheduling: true,
        currentEvent: {
          start,
          end,
          title: '',
        },
      });
    }
  };

  handleInputChange = (event) => {
    const { currentEvent } = this.state;
    this.setState({
      currentEvent: {
        ...currentEvent,
        title: event.target.value,
      },
    });
  };

  handleNext = () => {
    const { currentEvent } = this.state;
    if (currentEvent.title.trim() === '') {
      alert('Please enter an event title.');
    } else {
      this.setState((prevState) => ({
        events: [...prevState.events, currentEvent],
        isScheduling: false,
        currentEvent: {
          start: null,
          end: null,
          title: '',
        },
      }));
    }
  };

  render() {
    const { events, isScheduling, currentEvent } = this.state;

    return (
      <div>
        <h1>My Calendar</h1>
        <Calendar
          localizer={localizer}
          selectable
          events={events}
          onSelectSlot={this.handleSelectSlot}
          style={{ height: 500 }}
        />

        {isScheduling && (
          <div>
            <h2>Schedule Event</h2>
            <label>
              Event Title:
              <input
                type="text"
                value={currentEvent.title}
                onChange={this.handleInputChange}
              />
            </label>
            <button onClick={this.handleNext}>Next</button>
          </div>
        )}
        <label htmlFor="type">Select Vehicle Type:</label>
      <select id="type" value={selectedType} onChange={handleTypeChange}>
        <option value="">Select Type</option>
        {vehicleTypes.map((type) => (
          <option key={type} value={type}>
            {type}
          </option>
        ))}
      </select>

      {selectedType && (
        <div>
          <label htmlFor="vehicle">Select Vehicle:</label>
          <select id="vehicle">
            <option value="">Select Vehicle</option>
            {availableVehicles.map((vehicle) => (
              <option key={vehicle} value={vehicle}>
                {vehicle}
              </option>
            ))}
          </select>
        </div>
      )}
      </div>
    );
  }
}

export default MyCalendar;