#!/usr/bin/env python

import argparse

from scapy.all import load_layer
from scapy.sendrecv import AsyncSniffer

# if using pip package uncomment the following line
# from meter.flow_session import generate_session_class

from flow_session import generate_session_class
from doh_models.load_model import load_model
from doh_models.doh_model import DoH


def create_sniffer(input_file, input_interface, output_mode, output_file, is_nids_enabled, ml_model):
    assert (input_file is None) ^ (input_interface is None)

    NewFlowSession = generate_session_class(output_mode, output_file, is_nids_enabled, ml_model)

    if input_file is not None:
        # return AsyncSniffer(offline=input_file, filter='tcp port 443', prn=None, session=NewFlowSession, store=False)
        return AsyncSniffer(offline=input_file, filter='host 1.1.1.1 and tcp port 443', prn=None, session=NewFlowSession, store=False)
    else:
        # return AsyncSniffer(iface=input_interface, filter='tcp port 443', prn=None,
        return AsyncSniffer(iface=input_interface, filter='host 1.1.1.1 and tcp port 443', prn=None,
                            session=NewFlowSession, store=False)


def main():
    parser = argparse.ArgumentParser()

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-n', '--online', '--interface', action='store', dest='input_interface',
                             help='capture online data from INPUT_INTERFACE')
    input_group.add_argument('-f', '--offline', '--file', action='store', dest='input_file',
                             help='capture offline data from INPUT_FILE')

    # ML model options
    parser.add_argument('-m', '--model', action='store', dest='model_path',
                             help='load model MODEL_PATH')

    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument('-c', '--csv', '--flow', action='store_const', const='flow', dest='output_mode',
                              help='output flows as csv')
    output_group.add_argument('-s', '--json', '--sequence', action='store_const', const='sequence', dest='output_mode',
                              help='output flow segments as json')

    parser.add_argument('output', help='output file name (in flow mode) or directory (in sequence mode)')
    parser.add_argument('--enable-nids', action='store_true', help='Enable NIDS mode')
    args = parser.parse_args()

    doh_model = None
    load_layer('tls')
    if args.model_path and args.enable_nids:
        print("loading model...", args.model_path)
        # print("output directory...", args.output)
        doh_model = DoH(args.model_path)
        sniffer = create_sniffer(args.input_file, args.input_interface, args.output_mode, args.output, True, doh_model)

    else:
        sniffer = create_sniffer(args.input_file, args.input_interface, args.output_mode, args.output, False, doh_model)

    sniffer.start()

    try:
        sniffer.join()
    except KeyboardInterrupt:
        sniffer.stop()
    finally:
        sniffer.join()


if __name__ == '__main__':
    main()
