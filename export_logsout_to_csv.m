\
    %% Export logged Simulink signals to CSV
    % Before running: open the pneumatic model, enable signal logging for the
    % signals you want, and ensure the model logs to a variable named logsout.
    % This script does NOT invent sensor measurements or create fault labels.
    clearvars -except simOut
    if ~exist('simOut','var')
        error(['Run your Simulink model with signal logging enabled first, ', ...
               'then set simOut = sim(modelName) or adapt this script to your output.']);
    end
    if ~isprop(simOut,'logsout') && ~isfield(simOut,'logsout')
        error('No logsout found. Enable Signal logging in Simulink and rerun.');
    end
    logs = simOut.logsout;
    if isempty(logs) || logs.numElements == 0
        error('logsout is empty. Right-click each desired signal and select Log Selected Signals.');
    end
    T = table();
    for k = 1:logs.numElements
        el = logs.get(k);
        nm = matlab.lang.makeValidName(el.Name);
        val = el.Values;
        if isa(val,'timeseries')
            t = val.Time(:);
            y = squeeze(val.Data);
            if isvector(y)
                if isempty(T)
                    T.time_s = t;
                end
                T.(nm) = y(:);
            else
                warning('Skipping non-scalar signal "%s"; export its channels separately.', el.Name);
            end
        else
            warning('Skipping signal "%s": not a timeseries.', el.Name);
        end
    end
    if isempty(T)
        error('No scalar timeseries were exported. Inspect logsout signal names/types.');
    end
    writetable(T, fullfile(pwd,'simscape_sensor_export.csv'));
    fprintf('Exported %d rows and %d columns to simscape_sensor_export.csv\n', height(T), width(T));
