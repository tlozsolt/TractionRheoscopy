% Start Commumication with serial port
piezo_comm = serial('/dev/tty.usbserial','Baudrate',115200,'Stopbits',1,'Databits',8,'Terminator',10,'FlowControl','Hardware')
fopen(piezo_comm)
%
fprintf(piezo_comm,'*IDN?\n')
query=fscanf(piezo_comm,'%s') %query what piezo is connected?
sprintf('%s\n',query) %print 'query' with the format as a string
%
fprintf(piezo_comm,'SVO A1\n') % Turn the servo on
fprintf(piezo_comm,'SVO? A\n') %query "what's the state of the servo?"
fscanf(piezo_comm,'%f') %print reply, 1 means on, 0 means off
%
fprintf(piezo_comm,'MOV A%s\n','10') %Now move the piezo to position 10um
fprintf(piezo_comm,'POS? A\n') %query, "what is the position?"
fscanf(piezo_comm,'%f') % print reply
%
% Close sequence
fclose(piezo_comm) % close the file object
instrreset % clear all connected serial ports
clear piezo_comm % erase piezo_comm from namespace
%
% MANUALLY turn off piezo on switch in the back before unplugging anything.
%
% turn on piezo again and relauch the start sequence.