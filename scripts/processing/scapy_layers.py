
#!/usr/bin/python3
#-*-coding: utf-8-*-

from scapy.packet import *
from scapy.fields import *
from scapy.ansmachine import *
from scapy.layers.inet import *
import base64
import os
import string
import random


class IMAPField(StrField):
    """
    field class for handling imap packets
    @attention: it inherets StrField from Scapy library
    """

    def getfield(self, pkt, s):
        """
        this method will get the packet, takes what does need to be
        taken and let the remaining go, so it returns two values.
        first value which belongs to this field and the second is
        the remaining which does need to be dissected with
        other "field classes".
        @param pkt: holds the whole packet
        @param s: holds only the remaining data which is not dissected yet.
        """
        if (not isinstance(s, str)):
          # Convert to string
          s = s.decode("utf-8") 
        remain = ""
        value = ""
        ls = s.splitlines()
        myresult = ""
        lslen = len(ls)
        i = 0
        k = 0
        for line in ls:
            k = k + 1
            ls2 = line.split()
            length = len(ls2)
            if length > 1:
                value = ls2[0]
                c = 1
                remain = ""
                while c < length:
                    remain = remain + ls2[c] + " "
                    c = c + 1
                if self.name.startswith("request"):
                    myresult = myresult + "Request Tag: " +\
                            value + ", Request Argument: " + remain
                    #print('je passe : ', myresult)
                    if k < lslen:
                        myresult = myresult + " | "
                if self.name.startswith("response"):
                    myresult = myresult + "Response Tag: " +\
                    value + ", Response Argument: " + remain
                    if k < lslen:
                        myresult = myresult + " | "
            i = i + 1
            if i == lslen:
                return "", myresult

    def __init__(self, name, default, fmt, remain=0):
        """
        class constructor for initializing the instance variables
        @param name: name of the field
        @param default: Scapy has many formats to represent the data
        internal, human and machine. anyways you may sit this param to None.
        @param fmt: specifying the format, this has been set to "H"
        @param remain: this parameter specifies the size of the remaining
        data so make it 0 to handle all of the data.
        """
        self.name = name#"request"#"IMAPField"#name
        StrField.__init__(self, name, default, fmt, remain)


class IMAPRes(Packet):
    """
    class for handling imap responses
    @attention: it inherets Packet from Scapy library
    """
    name = "IMAP"
    fields_desc = [IMAPField("response", "", "H")]


class IMAPReq(Packet):
    """
    class for handling imap requests
    @attention: it inherets Packet from Scapy library
    """
    name = "IMAP"
    fields_desc = [IMAPField("request", "", "H")]


bind_layers(TCP, IMAPReq, dport=143)
bind_layers(TCP, IMAPRes, sport=143)

import base64
import base64
import os
import string
import random
from scapy.packet import *
from scapy.fields import *
from scapy.ansmachine import *
from scapy.layers.inet import *


class SMTPResField(StrField):
    """
    this is a field class for handling the smtp data
    @attention: this class inherets StrField
    """

    def get_code_msg(self, cn):
        """
        method returns a message for every a specific code number
        @param cn: code number
        """
        codes = {
                 "500": "Syntax error, command unrecognized",
                 "501": "Syntax error in parameters or arguments",
                 "502": "Command not implemented",
                 "503": "Bad sequence of commands",
                 "504": "Command parameter not implemented",
                 "211": "System status, or system help reply",
                 "214": "Help message",
                 "220": "<domain> Service ready",
                 "221": "<domain> Service closing transmission channel",
                 "421": "<domain> Service not available,\
                 closing transmission channel",
                 "250": "Requested mail action okay, completed",
                 "251": "User not local; will forward to <forward-path>",
                 "450": "Requested mail action not taken: mailbox unavailable",
                 "550": "Requested action not taken: mailbox unavailable",
                 "451": "Requested action aborted: error in processing",
                 "551": "User not local; please try <forward-path>",
                 "452": "Requested action not taken: insufficient system\
                  storage",
                 "552": "Requested mail action aborted: exceeded storage\
                  allocation",
                 "553": "Requested action not taken: mailbox name not allowed",
                 "354": "Start mail input; end with <CRLF>.<CRLF>",
                 "554": "Transaction failed",
                 "211": "System status, or system help reply",
                 "214": "Help message",
                 "220": "<domain> Service ready",
                 "221": "<domain> Service closing transmission channel",
                 "250": "Requested mail action okay, completed",
                 "251": "User not local; will forward to <forward-path>",
                 "354": "Start mail input; end with <CRLF>.<CRLF>",
                 "421": "<domain> Service not available, closing \
                 transmission channel",
                 "450": "Requested mail action not taken: mailbox unavailable",
                 "451": "Requested action aborted: local error in processing",
                 "452": "Requested action not taken: insufficient system\
                  storage",
                 "500": "Syntax error, command unrecognized",
                 "501": "Syntax error in parameters or arguments",
                 "502": "Command not implemented",
                 "503": "Bad sequence of commands",
                 "504": "Command parameter not implemented",
                 "550": "Requested action not taken: mailbox unavailable",
                 "551": "User not local; please try <forward-path>",
                 "552": "Requested mail action aborted: exceeded storage\
                  allocation",
                 "553": "Requested action not taken: mailbox name not allowed",
                 "554": "Transaction failed"}
        if cn in codes:
            return codes[cn]
        return "Unknown Response Code"

    def getfield(self, pkt, s):
        """
        this method will get the packet, takes what does need to be
        taken and let the remaining go, so it returns two values.
        first value which belongs to this field and the second is
        the remaining which does need to be dissected with
        other "field classes".
        @param pkt: holds the whole packet
        @param s: holds only the remaining data which is not dissected yet.
        """
        if (not isinstance(s, str)):
          # Convert to string
          s = s.decode("utf-8") 
        remain = ""
        value = ""
        ls = s.splitlines()
        length = len(ls)
        if length == 1:
            value = ls[0]
            arguments = ""
            first = True
            res = value.split(" ")
            for arg in res:
                if not first:
                    arguments = arguments + arg + " "
                first = False
            if "-" in res[0]:
                value = "(" + res[0][:3] + ") " +\
                 self.get_code_msg(res[0][:3]) + " " + res[0][3:]
            else:
                value = "(" + res[0] + ") " + self.get_code_msg(res[0])
            return arguments[:-1], [value]

        if length > 1:
            reponses = []
            for element in ls:
                element = element.split(" ")
                arguments = ""
                first = True
                for arg in element:
                    if not first:
                        arguments = arguments + arg + " "
                    first = False
                if "-" in element[0]:
                    reponses.append(["(" + element[0][:3] + ") " +
                                      self.get_code_msg(element[0][:3]) +
                                       " " + element[0][3:], arguments[:-1]])
                else:
                    reponses.append(["(" + element[0] + ") " +
                                      self.get_code_msg(element[0][:-1]),
                                       arguments])
            return "", reponses
        return "", ""

    def __init__(self, name, default, fmt, remain=0):
        """
        class constructor for initializing the instance variables
        @param name: name of the field
        @param default: Scapy has many formats to represent the data
        internal, human and machine. anyways you may sit this param to None.
        @param fmt: specifying the format, this has been set to "H"
        @param remain: this parameter specifies the size of the remaining
        data so make it 0 to handle all of the data.
        """
        self.name = name
        StrField.__init__(self, name, default, fmt, remain)


class SMTPReqField(StrField):

    def getfield(self, pkt, s):
        """
        this method will get the packet, takes what does need to be
        taken and let the remaining go, so it returns two values.
        first value which belongs to this field and the second is
        the remaining which does need to be dissected with
        other "field classes".
        @param pkt: holds the whole packet
        @param s: holds only the remaining data which is not dissected yet.
        """
        if (not isinstance(s, str)):
          # Convert to string
          s = s.decode("utf-8") 

        remain = ""
        value = ""
        ls = s.split()
        length = len(ls)

        if length > 1:
            value = ls[0]
            if length == 2:
                remain = ls[1]
                return remain, value
            else:
                i = 1
                remain = ' '
                while i < length:
                    remain = remain + ls[i] + ' '
                    i = i + 1
                return remain[:-1], value
        else:
            return "", ls[0]

    def __init__(self, name, default, fmt, remain=0):
        """
        class constructor for initializing the instance variables
        @param name: name of the field
        @param default: Scapy has many formats to represent the data
        internal, human and machine. anyways you may sit this param to None.
        @param fmt: specifying the format, this has been set to "H"
        @param remain: this parameter specifies the size of the remaining
        data so make it 0 to handle all of the data.
        """
        self.name = name
        StrField.__init__(self, name, default, fmt, remain)

class SMTPResponse(Packet):
    """
    class for handling the smtp responses
    @attention: this class inherets Packet
    """
    name = "SMTP"
    fields_desc = [SMTPResField("response", "", "H"),
                    StrField("argument", "", "H")]


class SMTPRequest(Packet):
    """
    class for handling the smtp requests
    @attention: this class inherets Packet
    """
    name = "SMTP"
    fields_desc = [SMTPReqField("command", '', "H"),
                    StrField("argument", '', "H")]

bind_layers(TCP, SMTPResponse, sport=25)
bind_layers(TCP, SMTPRequest, dport=25)
bind_layers(TCP, SMTPResponse, sport=587)
bind_layers(TCP, SMTPRequest, dport=587)

from scapy.packet import *
from scapy.fields import *
from scapy.ansmachine import *
from scapy.layers.inet import *


class POPField(StrField):
    """
    field class for handling pop requests
    @attention: it inherets StrField from Scapy library
    """

    def getfield(self, pkt, s):
        """
        this method will get the packet, takes what does need to be
        taken and let the remaining go, so it returns two values.
        first value which belongs to this field and the second is
        the remaining which does need to be dissected with
        other "field classes".
        @param pkt: holds the whole packet
        @param s: holds only the remaining data which is not dissected yet.
        """
        if (not isinstance(s, str)):
          # Convert to string
          s = s.decode("utf-8")
        remain = ""
        value = ""
        ls = s.splitlines()
        #myresult = []
        myresult = ""
        lslen = len(ls)
        i = 0
        k = 0
        for line in ls:
            k = k + 1
            ls2 = line.split()
            length = len(ls2)
            if length > 1:
                value = ls2[0]
                c = 1
                remain = ""
                while c < length:
                    remain = remain + ls2[c] + " "
                    c = c + 1
                if self.name.startswith("request"):
                    myresult = myresult + "Request Command: " + value +\
                    ", Request Parameter(s): " + remain
                    if k < lslen:
                        myresult = myresult + " | "
                if self.name.startswith("response"):
                    myresult = myresult + "Response Indicator: " + value +\
                    ", Response Parameter(s): " + remain
                    if k < lslen:
                        myresult = myresult + " | "
            i = i + 1
            if i == lslen:
                return "", myresult

    def __init__(self, name, default, fmt, remain=0):
        """
        class constructor for initializing the instance variables
        @param name: name of the field
        @param default: Scapy has many formats to represent the data
        internal, human and machine. anyways you may sit this param to None.
        @param fmt: specifying the format, this has been set to "H"
        @param remain: this parameter specifies the size of the remaining
        data so make it 0 to handle all of the data.
        """
        self.name = name
        StrField.__init__(self, name, default, fmt, remain)


class POPRes(Packet):
    """
    class for handling pop responses
    @attention: it inherets Packet from Scapy library
    """
    name = "POP"
    fields_desc = [POPField("response", "", "H")]


class POPReq(Packet):
    """
    class for handling pop requests
    @attention: it inherets Packet from Scapy library
    """
    name = "POP"
    fields_desc = [POPField("request", "", "H")]


bind_layers(TCP, POPReq, dport=110)
bind_layers(TCP, POPRes, sport=110)

import base64
from scapy.packet import *
from scapy.fields import *
from scapy.ansmachine import *
from scapy.layers.inet import *
from scapy.layers.dns import *

class SIPStartField(StrField):
    """
    field class for handling sip start field
    @attention: it inherets StrField from Scapy library
    """

    def getfield(self, pkt, s):
        """
        this method will get the packet, takes what does need to be
        taken and let the remaining go, so it returns two values.
        first value which belongs to this field and the second is
        the remaining which does need to be dissected with
        other "field classes".
        @param pkt: holds the whole packet
        @param s: holds only the remaining data which is not dissected yet.
        """
        if (not isinstance(s, str)):
          # Convert to string
          s = s.decode("utf-8") 
          
        remain = ""
        value = ""
        ls = s.splitlines(True)
        f = ls[0].split()
        if "SIP" in f[0]:
            ls = s.splitlines(True)
            f = ls[0].split()
            length = len(f)
            value = ""
            if length == 3:
                value = "SIP-Version:" + f[0] + ", Status-Code:" +\
                f[1] + ", Reason-Phrase:" + f[2]
                ls.remove(ls[0])
                for element in ls:
                    remain = remain + element
            else:
                value = ls[0]
                ls.remove(ls[0])
                for element in ls:
                    remain = remain + element
            return remain, value
        elif "SIP" in f[2]:
            ls = s.splitlines(True)
            f = ls[0].split()
            length = len(f)
            value = []
            if length == 3:
                value = "Method:" + f[0] + ", Request-URI:" +\
                f[1] + ", SIP-Version:" + f[2]
                ls.remove(ls[0])
                for element in ls:
                    remain = remain + element
            else:
                value = ls[0]
                ls.remove(ls[0])
                for element in ls:
                    remain = remain + element
            return remain, value
        else:
            return s, ""


class SIPMsgField(StrField):
    """
    field class for handling the body of sip packets
    @attention: it inherets StrField from Scapy library
    """

    def __init__(self, name, default):
        """
        class constructor, for initializing instance variables
        @param name: name of the field
        @param default: Scapy has many formats to represent the data
        internal, human and machine. anyways you may sit this param to None.
        """
        self.name = name
        self.fmt = "!B"
        Field.__init__(self, name, default, "!B")

    def getfield(self, pkt, s):
        """
        this method will get the packet, takes what does need to be
        taken and let the remaining go, so it returns two values.
        first value which belongs to this field and the second is
        the remaining which does need to be dissected with
        other "field classes".
        @param pkt: holds the whole packet
        @param s: holds only the remaining data which is not dissected yet.
        """
        if (not isinstance(s, str)):
          # Convert to string
          s = s.decode("utf-8") 
        if s.startswith("\r\n"):
            s = s.lstrip("\r\n")
            if s == "":
                return "", ""
        self.myresult = ""
        for c in s:
            self.myresult = self.myresult + base64.standard_b64encode(c)
        return "", self.myresult


class SIPField(StrField):
    """
    field class for handling the body of sip fields
    @attention: it inherets StrField from Scapy library
    """

    def getfield(self, pkt, s):
        """
        this method will get the packet, takes what does need to be
        taken and let the remaining go, so it returns two values.
        first value which belongs to this field and the second is
        the remaining which does need to be dissected with
        other "field classes".
        @param pkt: holds the whole packet
        @param s: holds only the remaining data which is not dissected yet.
        """
        if (not isinstance(s, str)):
          # Convert to string
          s = s.decode("utf-8") 

        if self.name == "unknown-header(s): ":
            remain = ""
            value = []
            ls = s.splitlines(True)
            i = -1
            for element in ls:
                i = i + 1
                if element == "\r\n":
                    return s, []
                elif element != "\r\n" and (": " in element[:10])\
                 and (element[-2:] == "\r\n"):
                    value.append(element)
                    ls.remove(ls[i])
                    remain = ""
                    unknown = True
                    for element in ls:
                        if element != "\r\n" and (": " in element[:15])\
                         and (element[-2:] == "\r\n") and unknown:
                            value.append(element)
                        else:
                            unknow = False
                            remain = remain + element
                    return remain, value
            return s, []

        remain = ""
        value = ""
        ls = s.splitlines(True)
        i = -1
        for element in ls:
            i = i + 1
            if element.upper().startswith(self.name.upper()):
                value = element
                value = value.strip(self.name)
                ls.remove(ls[i])
                remain = ""
                for element in ls:
                    remain = remain + element
                return remain, value[len(self.name) + 1:]
        return s, ""

    def __init__(self, name, default, fmt, remain=0):
        """
        class constructor for initializing the instance variables
        @param name: name of the field
        @param default: Scapy has many formats to represent the data
        internal, human and machine. anyways you may sit this param to None.
        @param fmt: specifying the format, this has been set to "H"
        @param remain: this parameter specifies the size of the remaining
        data so make it 0 to handle all of the data.
        """
        self.name = name
        StrField.__init__(self, name, default, fmt, remain)


class SIP(Packet):
    """
    class for handling the body of sip packets
    @attention: it inherets Packet from Scapy library
    """
    name = "SIP"
    fields_desc = [SIPStartField("start-line: ", "", "H"),
                   SIPField("accept: ", "", "H"),
                   SIPField("accept-contact: ", "", "H"),
                   SIPField("accept-encoding: ", "", "H"),
                   SIPField("accept-language: ", "", "H"),
                   SIPField("accept-resource-priority: ", "", "H"),
                   SIPField("alert-info: ", "", "H"),
                   SIPField("allow: ", "", "H"),
                   SIPField("allow-events: ", "", "H"),
                   SIPField("authentication-info: ", "", "H"),
                   SIPField("authorization: ", "", "H"),
                   SIPField("call-id: ", "", "H"),
                   SIPField("call-info: ", "", "H"),
                   SIPField("contact: ", "", "H"),
                   SIPField("content-disposition: ", "", "H"),
                   SIPField("content-encoding: ", "", "H"),
                   SIPField("content-language: ", "", "H"),
                   SIPField("content-length: ", "", "H"),
                   SIPField("content-type: ", "", "H"),
                   SIPField("cseq: ", "", "H"),
                   SIPField("date: ", "", "H"),
                   SIPField("error-info: ", "", "H"),
                   SIPField("event: ", "", "H"),
                   SIPField("expires: ", "", "H"),
                   SIPField("from: ", "", "H"),
                   SIPField("in-reply-to: ", "", "H"),
                   SIPField("join: ", "", "H"),
                   SIPField("max-forwards: ", "", "H"),
                   SIPField("mime-version: ", "", "H"),
                   SIPField("min-expires: ", "", "H"),
                   SIPField("min-se: ", "", "H"),
                   SIPField("organization: ", "", "H"),
                   SIPField("p-access-network-info: ", "", "H"),
                   SIPField("p-asserted-identity: ", "", "H"),
                   SIPField("p-associated-uri: ", "", "H"),
                   SIPField("p-called-party-id: ", "", "H"),
                   SIPField("p-charging-function-addresses: ", "", "H"),
                   SIPField("p-charging-vector: ", "", "H"),
                   SIPField("p-dcs-trace-party-id: ", "", "H"),
                   SIPField("p-dcs-osps: ", "", "H"),
                   SIPField("p-dcs-billing-info: ", "", "H"),
                   SIPField("p-dcs-laes: ", "", "H"),
                   SIPField("p-dcs-redirect: ", "", "H"),
                   SIPField("p-media-authorization: ", "", "H"),
                   SIPField("p-preferred-identity: ", "", "H"),
                   SIPField("p-visited-network-id: ", "", "H"),
                   SIPField("path: ", "", "H"),
                   SIPField("priority: ", "", "H"),
                   SIPField("privacy: ", "", "H"),
                   SIPField("proxy-authenticate: ", "", "H"),
                   SIPField("proxy-authorization: ", "", "H"),
                   SIPField("proxy-require: ", "", "H"),
                   SIPField("rack: ", "", "H"),
                   SIPField("reason: ", "", "H"),
                   SIPField("record-route: ", "", "H"),
                   SIPField("referred-by: ", "", "H"),
                   SIPField("reject-contact: ", "", "H"),
                   SIPField("replaces: ", "", "H"),
                   SIPField("reply-to: ", "", "H"),
                   SIPField("request-disposition: ", "", "H"),
                   SIPField("require: ", "", "H"),
                   SIPField("resource-priority: ", "", "H"),
                   SIPField("retry-after: ", "", "H"),
                   SIPField("route: ", "", "H"),
                   SIPField("rseq: ", "", "H"),
                   SIPField("security-client: ", "", "H"),
                   SIPField("security-server: ", "", "H"),
                   SIPField("security-verify: ", "", "H"),
                   SIPField("server: ", "", "H"),
                   SIPField("service-route: ", "", "H"),
                   SIPField("session-expires: ", "", "H"),
                   SIPField("sip-etag: ", "", "H"),
                   SIPField("sip-if-match: ", "", "H"),
                   SIPField("subject: ", "", "H"),
                   SIPField("subscription-state: ", "", "H"),
                   SIPField("supported: ", "", "H"),
                   SIPField("timestamp: ", "", "H"),
                   SIPField("to: ", "", "H"),
                   SIPField("unsupported: ", "", "H"),
                   SIPField("user-agent: ", "", "H"),
                   SIPField("via: ", "", "H"),
                   SIPField("warning: ", "", "H"),
                   SIPField("www-authenticate: ", "", "H"),
                   SIPField("refer-to: ", "", "H"),
                   SIPField("history-info: ", "", "H"),
                   SIPField("unknown-header(s): ", "", "H"),
                   SIPMsgField("message-body: ", "")]

bind_layers(TCP, SIP, sport=5060)
bind_layers(TCP, SIP, dport=5060)
bind_layers(UDP, SIP, sport=5060)
bind_layers(UDP, SIP, dport=5060)

import binascii
import base64
import json
from scapy.packet import *
from scapy.utils import *
from scapy.fields import *
from scapy.ansmachine import *
from scapy.layers.inet import *

def int2bin(n, count=16):
    """
    this method converts integer numbers to binary numbers
    @param n: the number to be converted
    @param count: the number of binary digits
    """
    return "".join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

# holds ssh encrypted sessions
encryptedsessions = []

class SSHField(XByteField): #XByteField
    """
    this is a field class for handling the ssh packets
    @attention: this class inherets XByteField
    """

    def get_ascii(self, hexstr):
        """
        get hex string and returns ascii chars
        @param hexstr: hex value in str format
        """
        return binascii.unhexlify(hexstr)

    def __init__(self, name, default):
        """
        class constructor, for initializing instance variables
        @param name: name of the field
        @param default: Scapy has many formats to represent the data
        internal, human and machine. anyways you may sit this param to None.
        """
        self.name = name
        self.fmt = "!B"
        Field.__init__(self, name, default, "!B")

    def get_discnct_msg(self, cn):
        """
        method returns a message for every a specific code number
        @param cn: code number
        """
        codes = {
                 1: "SSH_DISCONNECT_HOST_NOT_ALLOWED_TO_CONNECT",
                 2: "SSH_DISCONNECT_PROTOCOL_ERROR",
                 3: "SSH_DISCONNECT_KEY_EXCHANGE_FAILED",
                 4: "SSH_DISCONNECT_RESERVED",
                 5: "SSH_DISCONNECT_MAC_ERROR",
                 6: "SSH_DISCONNECT_COMPRESSION_ERROR",
                 7: "SSH_DISCONNECT_SERVICE_NOT_AVAILABLE",
                 8: "SSH_DISCONNECT_PROTOCOL_VERSION_NOT_SUPPORTED",
                 9: "SSH_DISCONNECT_HOST_KEY_NOT_VERIFIABLE",
                 10: "SSH_DISCONNECT_CONNECTION_LOST",
                 11: "SSH_DISCONNECT_BY_APPLICATION",
                 12: "SSH_DISCONNECT_TOO_MANY_CONNECTIONS",
                 13: "SSH_DISCONNECT_AUTH_CANCELLED_BY_USER",
                 14: "SSH_DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE",
                 15: "SSH_DISCONNECT_ILLEGAL_USER_NAME",
                 }
        if cn in codes:
            return codes[cn] + " "
        return "UnknownCode[" + str(cn) + "] "

    def get_code_msg(self, cn):
        """
        method returns a message for every a specific code number
        @param cn: code number
        """
        codes = {
                 1: "SSH_MSG_DISCONNECT",
                 2: "SSH_MSG_IGNORE",
                 3: "SSH_MSG_UNIMPLEMENTED",
                 4: "SSH_MSG_DEBUG",
                 5: "SSH_MSG_SERVICE_REQUEST",
                 6: "SSH_MSG_SERVICE_ACCEPT",
                 20: "SSH_MSG_KEXINIT",
                 21: "SSH_MSG_NEWKEYS",
                 30: "SSH_MSG_KEXDH_INIT",
                 31: "SSH_MSG_KEXDH_REPLY",
                 32: "SSH_MSG_KEX_DH_GEX_INIT",
                 33: "SSH_MSG_KEX_DH_GEX_REPLY",
                 34: "SSH_MSG_KEX_DH_GEX_REQUEST",
                 50: "SSH_MSG_USERAUTH_REQUEST",
                 51: "SSH_MSG_USERAUTH_FAILURE",
                 52: "SSH_MSG_USERAUTH_SUCCESS",
                 53: "SSH_MSG_USERAUTH_BANNER",
                 60: "SSH_MSG_USERAUTH_PK_OK",
                 80: "SSH_MSG_GLOBAL_REQUEST",
                 81: "SSH_MSG_REQUEST_SUCCESS",
                 82: "SSH_MSG_REQUEST_FAILURE",
                 90: "SSH_MSG_CHANNEL_OPEN",
                 91: "SSH_MSG_CHANNEL_OPEN_CONFIRMATION",
                 92: "SSH_MSG_CHANNEL_OPEN_FAILURE",
                 93: "SSH_MSG_CHANNEL_WINDOW_ADJUST",
                 94: "SSH_MSG_CHANNEL_DATA",
                 95: "SSH_MSG_CHANNEL_EXTENDED_DATA",
                 96: "SSH_MSG_CHANNEL_EOF",
                 97: "SSH_MSG_CHANNEL_CLOSE",
                 98: "SSH_MSG_CHANNEL_REQUEST",
                 99: "SSH_MSG_CHANNEL_SUCCESS",
                 100: "SSH_MSG_CHANNEL_FAILURE"}
        if cn in codes:
            return codes[cn] + " "
        return "UnknownCode[" + str(cn) + "] "

    def getfield(self, pkt, s):
        """
        this method will get the packet, takes what does need to be
        taken and let the remaining go, so it returns two values.
        first value which belongs to this field and the second is
        the remaining which does need to be dissected with
        other "field classes".
        @param pkt: holds the whole packet
        @param s: holds only the remaining data which is not dissected yet.
        """
        try:
          if (not isinstance(s, str)):
            # Convert to string
            s = s.decode("utf-8") # ISO-8859-1 # cp1252
        except:
          pass

        myresult = ""
        resultlist = []
        try:
          if s.upper().startswith("SSH"):
              return "", s
        except:
          pass

        for c in s:
            ustruct = struct.unpack(self.fmt, bytes([c]))
            byte = str(hex(ustruct[0]))[2:]
            if len(byte) == 1:
                byte = "0" + byte
            myresult = "" + byte
        return "", s

class SSH(Packet):
    """
    class for handling the ssh packets
    @attention: this class inherets Packet
    """
    name = "SSH"
    fields_desc = [SSHField("sshpayload", "")]

bind_layers(TCP, SSH, dport=22)
bind_layers(TCP, SSH, sport=22)

import base64
from scapy.packet import *
from scapy.utils import *
from scapy.fields import *
from scapy.ansmachine import *
from scapy.layers.inet import *

class TELNETField(XByteField):
    """
    field class for handling the telnet packets
    @attention: this class inherets XByteField
    """

    def __init__(self, name, default):
        """
        class constructor, for initializing instance variables
        @param name: name of the field
        @param default: Scapy has many formats to represent the data
        internal, human and machine. anyways you may sit this param to None.
        """
        self.name = name
        self.fmt = "!B"
        Field.__init__(self, name, default, "!B")

    def get_code_msg(self, cn):
        """
        method returns a message for every a specific code number
        @param cn: code number
        """
        codes = {0: "TRANSMIT-BINARY", 1: "ECHO",
                  3: "SUPPRESS-GO-AHEAD",
                  5: "STATUS", 6: "TIMING-MARK",
                   7: "RCTE", 10: "NAOCRD",
                  11: "NAOHTS", 12: "NAOHTD",
                   13: "NAOFFD", 14: "NAOVTS",
                  15: "NAOVTD", 16: "NAOLFD",
                   17: "EXTEND-ASCII",
                   18: "LOGOUT", 19: "BM", 20: "DET", 21: "SUPDUP",
                   22: "SUPDUP-OUTPUT", 23: "SEND-LOCATION",
                  24: "TERMINAL-TYPE", 25: "END-OF-RECORD",
                  26: "TUID", 27: "OUTMRK", 28: "TTYLOC", 29: "3270-REGIME",
                  30: "X.3-PAD", 31: "NAWS", 32: "TERMINAL-SPEED",
                  33: "TOGGLE-FLOW-CONTROL", 34: "LINEMODE",
                   35: "X-DISPLAY-LOCATION",
                  36: "ENVIRON", 37: "AUTHENTICATION", 38: "ENCRYPT",
                  39: "NEW-ENVIRON", 40: "TN3270E", 44: "COM-PORT-OPTION",
                  236: "End of Record", 237: "Suspend Current Process",
                  238: "Abort Process", 239: "End of File", 240: "SE",
                  241: "NOP", 242: "Data Mark", 243: "Break",
                  244: "Interrupt Process", 245: "Abort output",
                  246: "Are You There", 247: "Erase character",
                  248: "Erase Line", 249: "Go ahead", 250: "SB", 251: "WILL",
                  252: "WON'T", 253: "DO", 254: "DON'T", 255: "Command"}
        if cn in codes:
            return codes[cn] + " "
        return "UnknownCode[" + str(cn) + "] "

    def getfield(self, pkt, s):
        """
        this method will get the packet, takes what does need to be
        taken and let the remaining go, so it returns two values.
        first value which belongs to this field and the second is
        the remaining which does need to be dissected with
        other "field classes".
        @param pkt: holds the whole packet
        @param s: holds only the remaining data which is not dissected yet.
        """

        myresult = ""
        subOptions = False
        resultlist = []
        firstb = struct.unpack(self.fmt, bytes([s[0]]))[0]

        if firstb != 255:
            return "", s

        for c in s:
            ustruct = struct.unpack(self.fmt, bytes([c]))
            command = self.get_code_msg(ustruct[0])
            if command == "SB ":
                subOptions = True
                myresult = myresult + "SB "
                continue
            if command == "SE ":
                subOptions = False
                myresult = myresult = myresult + "SE "
                continue
            if subOptions:
                myresult = myresult +\
                 "subop(" + str(ustruct[0]) + ") "
                continue
            else:
                myresult = myresult + command
        # comlist = myresult.split("Command ")
        # for element in comlist:
        #     if element != "":
        #         resultlist.append(("command", element))
        #return  "", resultlist
        return  "", myresult


class TELNET(Packet):
    """
    field class for handling the telnet packets
    @attention: this class inherets Packet
    """
    name = "TELNET"
    fields_desc = [TELNETField("telnetpayload", "")]

bind_layers(TCP, TELNET, dport=23)
bind_layers(TCP, TELNET, sport=23)

class FTPResArgField(StrField):
    """
    class field to handle the ftp responses' arguments
    @attention: it inherets StrField which is imported from Scapy
    """

    def getfield(self, pkt, s):
        """
        this method will get the packet, takes what does need to be
        taken and let the remaining go, so it returns two values.
        first value which belongs to this field and the second is
        the remaining which does need to be dissected with
        other "field classes".
        @param pkt: holds the whole packet
        @param s: holds only the remaining data which is not dissected yet.
        """
        value = ""
        if "Entering Passive Mode (" in s:
            value = []
            res = s.split("Entering Passive Mode (")
            res.remove(res[0])
            res = res[0].split(").")
            del(res[len(res)-1])
            res = res[0].split(",")
            IP = res[0] + "." + res[1] + "." + res[2] + "." + res[3]
            Port = str(int(res[4]) * 256 + int(res[5]))
            value.append(("Passive IP Address", IP))
            value.append(("Passive Port Number", Port))
            return "", value
        else:
            value = s
            return "", value

    def __init__(self, name, default, fmt, remain=0):
        """
        FTPResArgField constructor for initializing the instance variables
        @param name: name of the field
        @param default: Scapy has many formats to represent the data
        internal, human and machine. anyways you may sit this param to None.
        @param fmt: specifying the format, this has been set to "H"
        @param remain: this parameter specifies the size of the remaining
        data so make it 0 to handle all of the data.
        """
        self.name = name
        StrField.__init__(self, name, default, fmt, remain)

class FTPDataField(XByteField):
    """
    this is a field class for handling the ftp data
    @attention: this class inherets XByteField
    """

    def __init__(self, name, default):
        """
        FTPDataField constructor, for initializing instance variables
        @param name: name of the field
        @param default: Scapy has many formats to represent the data
        internal, human and machine. anyways you may sit this param to None.
        """
        self.name = name
        Field.__init__(self, name, default, "!B")

    def getfield(self, pkt, s):
        return "", s


class FTPResField(StrField):
    """
    class field to handle the ftp responses
    @attention: it inherets StrField which is imported from Scapy
    """

    def get_code_msg(self, cn):
        """
        method which returns message for a ftp code number
        @param cn: code number
        """
        codes = {
    "110": "Restart marker reply",
    "120": "Service ready in nnn minutes",
    "125": "Data connection already open; transfer starting",
    "150": "File status okay; about to open data connection",
    "200": "Command okay",
    "202": "Command not implemented, superfluous at this site",
    "211": "System status, or system help reply",
    "212": "Directory status",
    "213": "File status",
    "214": "Help message",
    "215": "NAME system type",
    "220": "Service ready for new user",
    "221": "Service closing control connection",
    "225": "Data connection open; no transfer in progress",
    "226": "Closing data connection",
    "227": "Entering Passive Mode",
    "230": "User logged in proceed",
    "250": "Requested file action okay completed",
    "257": "PATHNAME created",
    "331": "User name okay need password",
    "332": "Need account for login",
    "350": "Requested file action pending further information",
    "421": "Service not available closing control connection",
    "425": "Can't open data connection",
    "426": "Connection closed; transfer aborted",
    "450": "Requested file action not taken",
    "451": "Requested action aborted: local error in processing",
    "452": "Requested action not taken. Insufficient storage space in system",
    "500": "Syntax error command unrecognized",
    "501": "Syntax error in parameters or arguments",
    "502": "Command not implemented",
    "503": "Bad sequence of commands",
    "504": "Command not implemented for that parameter",
    "530": "Not logged in",
    "532": "Need account for storing files",
    "550": "Requested action not taken: File unavailable",
    "551": "Requested action aborted: page type unknown",
    "552": "Requested file action aborted: Exceeded storage allocation",
    "553": "Requested action not taken: File name not allowed",
 }
        if cn in codes:
            return codes[cn]
        return ""

    def getfield(self, pkt, s):
        """
        this method will get the packet, takes what does need to be
        taken and let the remaining go, so it returns two values.
        first value which belongs to this field and the second is
        the remaining which does need to be dissected with
        other "field classes".
        @param pkt: holds the whole packet
        @param s: holds only the remaining data which is not dissected yet.
        """
        if (not isinstance(s, str)):
          # Convert to string
          s = s.decode("utf-8") 

        remain = ""
        value = ""
        ls = s.split()
        length = len(ls)
        if length > 1:
            value = self.get_code_msg(ls[0]) + " (" + ls[0] + ")"
            if length == 2:
                remain = ls[1]
                return remain, value
            else:
                i = 1
                remain = ""
                while i < length:
                    if i != 1:
                        remain = remain + " " + ls[i]
                    elif i == 1:
                        remain = remain + ls[i]
                    i = i + 1
                return remain, value
        else:
            return "", self.get_code_msg(ls[0]) + " (" + ls[0] + ")"

    def __init__(self, name, default, fmt, remain=0):
        """
        class constructor for initializing the instance variables
        @param name: name of the field
        @param default: Scapy has many formats to represent the data
        internal, human and machine. anyways you may sit this param to None.
        @param fmt: specifying the format, this has been set to "H"
        @param remain: this parameter specifies the size of the remaining
        data so make it 0 to handle all of the data.
        """
        self.name = name
        StrField.__init__(self, name, default, fmt, remain)


class FTPReqField(StrField):

    def getfield(self, pkt, s):
        """
        this method will get the packet, takes what does need to be
        taken and let the remaining go, so it returns two values.
        first value which belongs to this field and the second is
        the remaining which does need to be dissected with
        other "field classes".
        @param pkt: holds the whole packet
        @param s: holds only the remaining data which is not dissected yet.
        """
        if (not isinstance(s, str)):
          # Convert to string
          s = s.decode("utf-8") 

        remain = ""
        value = ""
        ls = s.split()
        if ls[0].lower() == "retr":
            c = 1
            file = ""
            while c < len(ls):
                file = file + ls[c]
                c = c + 1
            if len(file) > 0:
                add_file(file)
        length = len(ls)
        if length > 1:
            value = ls[0]
            if length == 2:
                remain = ls[1]
                return remain, value
            else:
                i = 1
                remain = ""
                while i < length:
                    remain = remain + ls[i] + " "
                    i = i + 1
                return remain, value
        else:
            return "", ls[0]

    def __init__(self, name, default, fmt, remain=0):
        """
        class constructor for initializing the instance variables
        @param name: name of the field
        @param default: Scapy has many formats to represent the data
        internal, human and machine. anyways you may sit this param to None.
        @param fmt: specifying the format, this has been set to "H"
        @param remain: this parameter specifies the size of the remaining
        data so make it 0 to handle all of the data.
        """
        self.name = name
        StrField.__init__(self, name, default, fmt, remain)


class FTPData(Packet):
    """
    class for dissecting the ftp data
    @attention: it inherets Packet class from Scapy library
    """
    name = "FTP"
    fields_desc = [FTPDataField("data", "")]


class FTPResponse(Packet):
    """
    class for dissecting the ftp responses
    @attention: it inherets Packet class from Scapy library
    """
    name = "FTP"
    fields_desc = [FTPResField("command", "", "H"),
                    FTPResArgField("argument", "", "H")]


class FTPRequest(Packet):
    """
    class for dissecting the ftp requests
    @attention: it inherets Packet class from Scapy library
    """
    name = "FTP"
    fields_desc = [FTPReqField("command", "", "H"),
                    StrField("argument", "", "H")]

bind_layers(TCP, FTPResponse, sport=21)
bind_layers(TCP, FTPRequest, dport=21)
bind_layers(TCP, FTPData, dport=20)
bind_layers(TCP, FTPData, dport=20)

from scapy.packet import *
from scapy.fields import *
from scapy.ansmachine import *
from scapy.layers.inet import *


class IRCResField(StrField):
    """
    field class for handling irc responses
    @attention: it inherets StrField from Scapy library
    """

    def getfield(self, pkt, s):
        """
        this method will get the packet, takes what does need to be
        taken and let the remaining go, so it returns two values.
        first value which belongs to this field and the second is
        the remaining which does need to be dissected with
        other "field classes".
        @param pkt: holds the whole packet
        @param s: holds only the remaining data which is not dissected yet.
        """
        if (not isinstance(s, str)):
          # Convert to string
          s = s.decode("utf-8")

        value = ""
        ls = s.split("\r\n")
        length = len(ls)
        if length == 1:
            return "", value
        elif length > 1:
                value = ""
                value = value + "response: " + ls[0]
                i = 1
                while i < length - 1:
                    value = value + " response: " + ls[i]
                    if i < length - 2:
                        value = value + " | "
                    i = i + 1
                return "", value
        else:
            return "", ""

    def __init__(self, name, default, fmt, remain=0):
        """
        class constructor for initializing the instance variables
        @param name: name of the field
        @param default: Scapy has many formats to represent the data
        internal, human and machine. anyways you may sit this param to None.
        @param fmt: specifying the format, this has been set to "H"
        @param remain: this parameter specifies the size of the remaining
        data so make it 0 to handle all of the data.
        """
        self.name = name
        StrField.__init__(self, name, default, fmt, remain)


class IRCReqField(StrField):
    """
    field class for handling irc requests
    @attention: it inherets StrField from Scapy library
    """

    def getfield(self, pkt, s):
        """
        this method will get the packet, takes what does need to be
        taken and let the remaining go, so it returns two values.
        first value which belongs to this field and the second is
        the remaining which does need to be dissected with
        other "field classes".
        @param pkt: holds the whole packet
        @param s: holds only the remaining data which is not dissected yet.
        """
        if (not isinstance(s, str)):
          # Convert to string
          s = s.decode("utf-8")

        remain = ""
        value = ""
        ls = s.split()
        length = len(ls)
        if length > 1:
            value = "command: " + ls[0] + ","
            if length == 2:
                remain = ls[1]
                value = value + " Parameters: " + remain
                return "", value
            else:
                i = 1
                remain = ""
                while i < length:
                    if i != 1:
                        remain = remain + " " + ls[i]
                    else:
                        remain = remain + ls[i]
                    i = i + 1
                value = value + " Parameters: " + remain
                return "", value
        else:
            return "", ls[0]

    def __init__(self, name, default, fmt, remain=0):
        """
        class constructor for initializing the instance variables
        @param name: name of the field
        @param default: Scapy has many formats to represent the data
        internal, human and machine. anyways you may sit this param to None.
        @param fmt: specifying the format, this has been set to "H"
        @param remain: this parameter specifies the size of the remaining
        data so make it 0 to handle all of the data.
        """
        self.name = name
        StrField.__init__(self, name, default, fmt, remain)


class IRCRes(Packet):
    """
    class for handling irc responses
    @attention: it inherets Packet from Scapy library
    """
    name = "IRC"
    fields_desc = [IRCResField("response", "", "H")]


class IRCReq(Packet):
    """
    class for handling irc requests
    @attention: it inherets Packet from Scapy library
    """
    name = "IRC"
    fields_desc = [IRCReqField("command", "", "H")]

bind_layers(TCP, IRCReq, dport=6660)
bind_layers(TCP, IRCReq, dport=6661)
bind_layers(TCP, IRCReq, dport=6662)
bind_layers(TCP, IRCReq, dport=6663)
bind_layers(TCP, IRCReq, dport=6664)
bind_layers(TCP, IRCReq, dport=6665)
bind_layers(TCP, IRCReq, dport=6666)
bind_layers(TCP, IRCReq, dport=6667)
bind_layers(TCP, IRCReq, dport=6668)
bind_layers(TCP, IRCReq, dport=6669)
bind_layers(TCP, IRCReq, dport=7000)
bind_layers(TCP, IRCReq, dport=194)
bind_layers(TCP, IRCReq, dport=6697)

bind_layers(TCP, IRCRes, sport=6660)
bind_layers(TCP, IRCRes, sport=6661)
bind_layers(TCP, IRCRes, sport=6662)
bind_layers(TCP, IRCRes, sport=6663)
bind_layers(TCP, IRCRes, sport=6664)
bind_layers(TCP, IRCRes, sport=6665)
bind_layers(TCP, IRCRes, sport=6666)
bind_layers(TCP, IRCRes, sport=6667)
bind_layers(TCP, IRCRes, sport=6668)
bind_layers(TCP, IRCRes, sport=6669)
bind_layers(TCP, IRCRes, sport=7000)
bind_layers(TCP, IRCRes, sport=194)
bind_layers(TCP, IRCRes, sport=6697)
