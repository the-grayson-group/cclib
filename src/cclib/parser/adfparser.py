"""
cclib is a parser for computational chemistry log files.

See http://cclib.sf.net for more information.

Copyright (C) 2006 Noel O'Boyle and Adam Tenderholt

 This program is free software; you can redistribute and/or modify it
 under the terms of the GNU General Public License as published by the
 Free Software Foundation; either version 2, or (at your option) any later
 version.

 This program is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY, without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 General Public License for more details.

Contributions (monetary as well as code :-) are encouraged.
"""
import re,time
import Numeric
import random # For sometimes running the progress updater
from logfileparser import Logfile,convertor

class ADF(Logfile):
    """A Gaussian 98/03 log file"""
    #SCFRMS,SCFMAX,SCFENERGY = range(3) # Used to index self.scftargets[]
    SCFCNV,SCFCNV2 = range(2) #used to index self.scftargets[]
    def __init__(self,*args):

        # Call the __init__ method of the superclass
        super(ADF, self).__init__(logname="ADF",*args)
        
    def __str__(self):
        """Return a string representation of the object."""
        return "ADF log file %s" % (self.filename)

    def __repr__(self):
        """Return a representation of the object."""
        return 'ADF("%s")' % (self.filename)

    def normalisesym(self,label):
        """Use standard symmetry labels instead of ADF labels.

        To normalise:
        (1) any periods are removed (except in the case of greek letters)
        (2) XXX is replaced by X, and a " added.
        (3) XX is replaced by X, and a ' added.
        (4) The greek letters Sigma, Pi, Delta and Phi are replaced by
            their lowercase equivalent.

        >>> sym = ADF("dummyfile").normalisesym
        >>> labels = ['A','s','A1','A1.g','Sigma','Pi','Delta','Phi','Sigma.g','A.g','AA','AAA','EE1','EEE1']
        >>> map(sym,labels)
        ['A', 's', 'A1', 'A1g', 'sigma', 'pi', 'delta', 'phi', 'sigma.g', 'Ag', "A'", 'A"', "E1'", 'E1"']
        """
        greeks = ['Sigma','Pi','Delta','Phi']
        for greek in greeks:
            if label.startswith(greek):
                return label.lower()
            
        ans = label.replace(".","")
        l = len(ans)
        if l>1 and ans[0]==ans[1]: # Python only tests the second condition if the first is true
            if l>2 and ans[1]==ans[2]:
                ans = ans.replace(ans[0]*3,ans[0]) + '"'
            else:
                ans = ans.replace(ans[0]*2,ans[0]) + "'"
        return ans
        

    def parse(self,fupdate=0.05,cupdate=0.002):
        """Extract information from the logfile."""
        inputfile = open(self.filename,"r")
        
        if self.progress:
            
            inputfile.seek(0,2) #go to end of file
            nstep=inputfile.tell()
            inputfile.seek(0)
            self.progress.initialize(nstep)
            oldstep=0
            
        for line in inputfile:
            
            if self.progress and random.random()<cupdate:
                step = inputfile.tell()
                if step!=oldstep:
                    self.progress.update(step,"Unsupported Information")
                    oldstep = step
                
            if line.find("INPUT FILE")>=0:
#check to make sure we aren't parsing Create jobs
                while line:
                
                    if self.progress and random.random()<fupdate:
                      step = inputfile.tell()
                      #if step!=oldstep:
                      self.progress.update(step,"Unsupported Information")
                      oldstep = step
                  
                    if line.find("INPUT FILE")>=0:
                      line2=inputfile.next()
                    if line2.find("Create")<0:
                      break
                            
                    line=inputfile.next()
            
            if line[1:6]=="ATOMS":
# Find the number of atoms and their atomic numbers
                if self.progress and random.random()<cupdate:
                    step=inputfile.tell()
                    if step!=oldstep:
                        self.progress.update(step,"Attributes")
                        oldstep=step
                
                self.logger.info("Creating attribute atomnos[]")
                self.atomnos=[]
                
                underline=inputfile.next()  #clear pointless lines
                label1=inputfile.next()     # 
                label2=inputfile.next()     #
                line=inputfile.next()
                while len(line)>1: #ensure that we are reading no blank lines
                    info=line.split()
                    self.atomnos.append(self.table.number[info[1]])
                    line=inputfile.next()
                
                self.natom=len(self.atomnos)
                self.logger.info("Creating attribute natom: %d" % self.natom)
                
            if line[1:22]=="S C F   U P D A T E S":
# find targets for SCF convergence (QM calcs)

              if not hasattr(self,"scftargets"):
                self.logger.info("Creating attribute scftargets[]")
              self.scftargets = Numeric.array([0.0, 0.0],'f')
              
              #underline, blank, nr
              for i in range(3): inputfile.next()
              
              line=inputfile.next()
              self.scftargets[ADF.SCFCNV]=float(line.split()[2])
              line=inputfile.next()
              self.scftargets[ADF.SCFCNV2]=float(line.split()[2])
              
            if line[1:11]=="CYCLE    1":
              
              if self.progress and random.random() < fupdate:
                step=inputfile.tell()
                if step!=oldstep:
                  self.progress.update(step, "QM Convergence")
                  oldstep=step
              
              if not hasattr(self,"scfvalues"):
                self.logger.info("Creating attribute scfvalues")
                self.scfvalues = []
                
              newlist = [ [] for x in self.scftargets ]
              line=inputfile.next()
              
              while line.find("SCF CONVERGED")==-1:
              
                if line[1:7]=="d-Pmat":
                  info=line.split()
                  newlist[ADF.SCFCNV].append(float(info[2]))
                  
                  line=inputfile.next()
                  info=line.split()
                  newlist[ADF.SCFCNV2].append(float(info[2]))
              
                try:
                  line=inputfile.next()
                except StopIteration: #EOF reached?
                  break
              
              self.scfvalues.append(newlist)
              
#             if line[1:10]=='Cycle   1':
# # Extract SCF convergence information (QM calcs)
#                 if self.progress and random.random()<fupdate:
#                     step=inputfile.tell()
#                     if step!=oldstep:
#                         self.progress.update(step,"QM Convergence")
#                         oldstep=step
#                         
#                 if not hasattr(self,"scfvalues"):
#                     self.logger.info("Creating attribute scfvalues")
#                     self.scfvalues = []
#                 newlist = [ [] for x in self.scftargets ]
#                 line = inputfile.next()
#                 while line.find("SCF Done")==-1:
#                     if line.find(' E=')==0:
#                         self.logger.debug(line)
#                     if line.find(" RMSDP")==0:
#                         parts = line.split()
#                         newlist[G03.SCFRMS].append(self.float(parts[0].split('=')[1]))
#                         newlist[G03.SCFMAX].append(self.float(parts[1].split('=')[1]))
#                         energy = 1.0
#                         if len(parts)>4:
#                             energy = parts[2].split('=')[1]
#                             if energy=="":
#                                 energy = self.float(parts[3])
#                             else:
#                                 energy = self.float(energy)
#                         # I moved the following line back a TAB to see the effect
#                         # (it was originally part of the above "if len(parts)")
#                         newlist[G03.SCFENERGY].append(energy)
#                     try:
#                         line = inputfile.next()
#                     except StopIteration: # May be interupted by EOF
#                         break
#                 self.scfvalues.append(newlist)
# 

            if line[1:27]=='Geometry Convergence Tests':
# Extract Geometry convergence information
                if not hasattr(self,"geotargets"):
                    self.logger.info("Creating attributes geotargets[],geovalues[[]]")
                    self.geovalues = []
                    self.geotargets = Numeric.array( [0.0,0.0,0.0,0.0,0.0],"f")
                if not hasattr(self,"scfenergies"):
                    self.logger.info("Creating attribute scfenergies[]")
                    self.scfenergies = []
                equals = inputfile.next()
                blank = inputfile.next()
                line = inputfile.next()
                temp = inputfile.next().strip().split()
                self.scfenergies.append(convertor(float(temp[-1]),"hartree","eV"))
                for i in range(6):
                    line = inputfile.next()
                values = []
                for i in range(5):
                    temp = inputfile.next().split()
                    self.geotargets[i] = float(temp[-3])
                    values.append(float(temp[-4]))
                self.geovalues.append(values)
 
            if line[1:29]=='Orbital Energies, all Irreps' and not hasattr(self,"mosyms"):
#Extracting orbital symmetries and energies, homos
              self.logger.info("Creating attribute mosyms[[]]")
              self.mosyms=[[]]
              
              self.logger.info("Creating attribute moenergies[[]]")
              self.moenergies=[[]]
              
              underline=inputfile.next()
              blank=inputfile.next()
              header=inputfile.next()
              underline2=inputfile.next()
              line=inputfile.next()
              
              homoa=None
              homob=None

              while len(line)==77:
                info=line.split()
                if len(info)==5: #this is restricted
                  self.mosyms[0].append(self.normalisesym(info[0]))
                  self.moenergies[0].append(convertor(float(info[3]),'hartree','eV'))
                  if info[2]=='0.00' and not hasattr(self,'homos'):
                      self.logger.info("Creating attribute homos[]")
                      self.homos=[len(self.moenergies[0])-2]
                  line=inputfile.next()
                elif len(info)==6: #this is unrestricted
                  if len(self.moenergies)<2: #if we don't have space, create it
                    self.moenergies.append([])
                    self.mosyms.append([])
                  if info[2]=='A':
                    self.mosyms[0].append(self.normalisesym(info[0]))
                    self.moenergies[0].append(convertor(float(info[4]),'hartree','eV'))
                    if info[3]=='0.00' and homoa==None:
                      homoa=len(self.moenergies[0])-2
                      
                  if info[2]=='B':
                    self.mosyms[1].append(self.normalisesym(info[0]))
                    self.moenergies[1].append(convertor(float(info[4]),'hartree','eV'))
                    if info[3]=='0.00' and homob==None:
                      homob=len(self.moenergies[1])-2
                      
                  line=inputfile.next()
                  
                else: #different number of lines
                  print "Error",info

              if len(info)==6: #still unrestricted, despite being out of loop
                self.logger.info("Creating attribute homos[]")
                self.homos=[homoa,homob]

#                tempa=Numeric.array(self.moenergies[0],"f")
#                tempb=Numeric.array(self.moenergies[1],"f")
#                self.moenergies=[tempa,tempb]
#              elif len(info)==5:
#                self.moenergies=[

              temp=Numeric.array(self.moenergies,"f")
              self.moenergies=temp

            if line[1:24]=="List of All Frequencies":
# Start of the IR/Raman frequency section
                if self.progress and random.random()<fupdate:
                    step=inputfile.tell()
                    if step!=oldstep:
                        self.progress.update(step,"Frequency Information")
                        oldstep=step
                         
#                 self.vibsyms = [] # Need to look into this a bit more
                self.vibirs = []
                self.vibfreqs = []
#                 self.logger.info("Creating attribute vibsyms[]")
                self.logger.info("Creating attribute vibfreqs[],vibirs[]")
                for i in range(8):
                    line = inputfile.next()
                line = inputfile.next().strip()
                while line:
                    temp = line.split()
                    self.vibfreqs.append(float(temp[0]))                    
                    self.vibirs.append(float(temp[2])) # or is it temp[1]?
                    line = inputfile.next().strip()
                self.vibfreqs = Numeric.array(self.vibfreqs,"f")
                self.vibirs = Numeric.array(self.vibirs,"f")
                if hasattr(self,"vibramans"): self.vibramans = Numeric.array(self.vibramans,"f")

#             if line[1:14]=="Excited State":
# # Extract the electronic transitions
#                 if not hasattr(self,"etenergy"):
#                     self.etenergies = []
#                     self.etoscs = []
#                     self.etsyms = []
#                     self.etsecs = []
#                     self.logger.info("Creating attributes etenergies[], etoscs[], etsyms[], etsecs[]")
#                 # Need to deal with lines like:
#                 # (restricted calc)
#                 # Excited State   1:   Singlet-BU     5.3351 eV  232.39 nm  f=0.1695
#                 # (unrestricted calc) (first excited state is 2!)
#                 # Excited State   2:   ?Spin  -A      0.1222 eV 10148.75 nm  f=0.0000
#                 parts = line[36:].split()
#                 self.etenergies.append(convertor(self.float(parts[0]),"eV","cm-1"))
#                 self.etoscs.append(self.float(parts[4].split("=")[1]))
#                 self.etsyms.append(line[21:36].split())
#                 
#                 line = inputfile.next()
# 
#                 p = re.compile("(\d+)")
#                 CIScontrib = []
#                 while line.find(" ->")>=0: # This is a contribution to the transition
#                     parts = line.split("->")
#                     self.logger.debug(parts)
#                     # Has to deal with lines like:
#                     #       32 -> 38         0.04990
#                     #      35A -> 45A        0.01921
#                     frommoindex = 0 # For restricted or alpha unrestricted
#                     fromMO = parts[0].strip()
#                     if fromMO[-1]=="B":
#                         frommoindex = 1 # For beta unrestricted
#                     fromMO = int(p.match(fromMO).group()) # extract the number
#                     
#                     t = parts[1].split()
#                     tomoindex = 0
#                     toMO = t[0]
#                     if toMO[-1]=="B":
#                         tomoindex = 1
#                     toMO = int(p.match(toMO).group())
# 
#                     percent = self.float(t[1])
#                     sqr = percent**2*2 # The fractional contribution of this CI
#                     if percent<0:
#                         sqr = -sqr
#                     CIScontrib.append([(fromMO,frommoindex),(toMO,tomoindex),sqr])
#                     line = inputfile.next()
#                 self.etsecs.append(CIScontrib)
#                 self.etenergies = Numeric.array(self.etenergies,"f")
#                 self.etoscs = Numeric.array(self.etoscs,"f")
# 
#             if line[1:52]=="<0|r|b> * <b|rxdel|0>  (Au), Rotatory Strengths (R)":
# # Extract circular dichroism data
#                 self.etrotats = []
#                 self.logger.info("Creating attribute etrotats[]")
#                 inputfile.next()
#                 inputfile.next()
#                 line = inputfile.next()
#                 parts = line.strip().split()
#                 while len(parts)==5:
#                     try:
#                         R = self.float(parts[-1])
#                     except ValueError:
#                         # nan or -nan if there is no first excited state
#                         # (for unrestricted calculations)
#                         pass
#                     else:
#                         self.etrotats.append(R)
#                     line = inputfile.next()
#                     temp = line.strip().split()
#                     parts = line.strip().split()                
#                 self.etrotats = Numeric.array(self.etrotats,"f")
# 

#******************************************************************************************************************8
#delete this after new implementation using smat, eigvec print,eprint?
            if line[1:49] == "Total nr. of (C)SFOs (summation over all irreps)":
# Extract the number of basis sets
              self.nbasis=int(line.split(":")[1].split()[0])
              self.logger.info("Creating attribute nbasis: %i" % self.nbasis)
                 
 # now that we're here, let's extract aonames
 
              self.logger.info("Creating attribute fonames[]")
              self.fonames=[]
                 
              blank=inputfile.next()
              note=inputfile.next()
              symoffset=0
              blank=inputfile.next(); blank=inputfile.next(); blank=inputfile.next()
              
              nosymreps=[]
              while len(self.fonames)<self.nbasis:
                          
                  sym=inputfile.next()
                  line=inputfile.next()
                  num=int(line.split(':')[1].split()[0])
                  nosymreps.append(num)
                     
                  #read until line "--------..." is found
                  while line.find('-----')<0:
                      line=inputfile.next()
                     
                  #for i in range(num):
                  while len(self.fonames)<symoffset+num:
                    line=inputfile.next()
                    info=line.split()
                      
                    #index0 index1 occ2 energy3/4 fragname5 coeff6 orbnum7 orbname8 fragname9
                    orbname=info[8]
                    orbital=info[7]+orbname.replace(":","")
                      
                    fragname=info[5]
                    frag=fragname+info[9]
                      
                    coeff=float(info[6])
                    if coeff**2<1.0: #is this a linear combination?
                      line=inputfile.next()
                      info=line.split()
                        
                      if line[42]==' ': #no new fragment type
                        frag+="+"+fragname+info[6]
                        coeff=float(info[3])
                        if coeff<0: orbital+='-'+info[4]+info[5].replace(":","")
                        else: orbital+='+'+info[4]+info[5].replace(":","")
                      else:
                        frag+="+"+info[3]+info[7]
                        coeff=float(info[4])
                        if coeff<0: orbital+='-'+info[5]+info[6].replace(":","")
                        else: orbital+="+"+info[5]+info[6].replace(":","")
                    
                    else:
                      inputfile.next()
                    self.fonames.append("%s_%s"%(frag,orbital))
                  symoffset+=num
                  
                  #nextline blankline blankline
                  inputfile.next(); inputfile.next(); inputfile.next()
                  
                  
#             if line[1:7]=="NBasis" or line[4:10]=="NBasis":
# # Extract the number of basis sets
#                 nbasis = int(line.split('=')[1].split()[0])
#                 # Has to deal with lines like:
#                 #  NBasis =   434 NAE=    97 NBE=    97 NFC=    34 NFV=     0
#                 #     NBasis = 148  MinDer = 0  MaxDer = 0
#                 # Although the former is in every file, it doesn't occur before
#                 # the overlap matrix is printed
#                 if hasattr(self,"nbasis"):
#                     assert nbasis==self.nbasis
#                 else:
#                     self.nbasis= nbasis
#                     self.logger.info("Creating attribute nbasis: %d" % self.nbasis)
#                     
#             if line[1:7]=="NBsUse":
# # Extract the number of linearly-independent basis sets
#                 nindep = int(line.split('=')[1].split()[0])
#                 if hasattr(self,"nindep"):
#                     assert nindep==self.nindep
#                 else:
#                     self.nindep = nindep
#                     self.logger.info("Creating attribute nindep: %d" % self.nindep)
# 
#             if line[7:22]=="basis functions,":
# # For AM1 calculations, set nbasis by a second method
# # (nindep may not always be explicitly stated)
#                     nbasis = int(line.split()[0])
#                     if hasattr(self,"nbasis"):
#                         assert nbasis==self.nbasis
#                     else:
#                         self.nbasis = nbasis
#                         self.logger.info("Creating attribute nbasis: %d" % self.nbasis)

#           
            if line[1:32]=="S F O   P O P U L A T I O N S ,":
#Extract overlap matrix

              self.logger.info("Creating attribute fooverlaps[x,y]")
              self.fooverlaps = Numeric.zeros((self.nbasis,self.nbasis),"float")
              
              symoffset=0
              
              for nosymrep in nosymreps:
                          
                line=inputfile.next()
                while line.find('===')<10: #look for the symmetry labels
                  line=inputfile.next()
                #blank blank text blank col row
                for i in range(6): inputfile.next()
                
                base=0
                
                while base<nosymrep: #have we read all the columns?
                      
                  for i in range(nosymrep-base):
                  
                    if self.progress:
                      step=inputfile.tell()
                      if step!=oldstep and random.random() < fupdate:
                        self.progress.update(step,"Overlap")
                        oldstep=step
                    
                    line=inputfile.next()
                    parts=line.split()[1:]
                    
                    for j in range(len(parts)):
                      k=float(parts[j])
                      self.fooverlaps[base+symoffset+j, base+symoffset+i] = k
                      self.fooverlaps[base+symoffset+i, base+symoffset+j] = k
                    
                  #blank, blank, column
                  for i in range(3): inputfile.next()
                  
                  base+=4
                                  
                symoffset+=nosymrep
                base=0
                    
            if line[48:67]=="SFO MO coefficients":
#extract MO coefficients              
              #read stars and three blank lines
              inputfile.next()
              inputfile.next()
              inputfile.next()
              inputfile.next()
              
              line=inputfile.next()
              
              if line.find("***** SPIN 1 *****")>0:
                beta = 1
                self.mocoeffs = Numeric.zeros((2,self.nbasis,self.nbasis),"float")
                
                #get rid of two blank lines and symmetry label
                inputfile.next()
                inputfile.next()
                sym=inputfile.next()
                #print sym
                
              else:
                beta = 0
                self.mocoeffs = Numeric.zeros((1,self.nbasis,self.nbasis),"float")
                
              #get rid of 12 lines of text
              for i in range(10):
                inputfile.next()
              
              for spin in range(beta+1):
                symoffset=0
                base=0
                
                if spin==1:
                  #read spin, blank, blank, symlabel, blank, text, underline, blank
                  for i in range(8):
                    line=inputfile.next()
                        
                while symoffset+base<self.nbasis:
            
                  line=inputfile.next()
                  if len(line)<3:
                    symoffset+=base
                    base=0
                    #print symoffset
                    
                  monumbers=line.split()
                  #print monumbers
                  #get rid of next two lines
                  inputfile.next()
                  inputfile.next()
                  
                  row=0
                  line=inputfile.next()
                  while len(line)>2:
                     
                    if self.progress:
                      step=inputfile.tell()
                      if step!=oldstep and random.random() < fupdate:
                        self.progress.update(step,"Coefficients")
                        oldstep=step

                    cols=line.split()
                    for i in range(len(cols[1:])):
                      self.mocoeffs[spin,row+symoffset,i+symoffset+base]=float(cols[i+1])
                  
                    line=inputfile.next()
                    row+=1
                  
                  base+=len(cols[1:])
                  
                  
#             if line[5:35]=="Molecular Orbital Coefficients" or line[5:41]=="Alpha Molecular Orbital Coefficients" or line[5:40]=="Beta Molecular Orbital Coefficients":
#                 if line[5:40]=="Beta Molecular Orbital Coefficients":
#                     beta = True
#                     # Need to add an extra dimension to self.mocoeffs
#                     self.mocoeffs = Numeric.resize(self.mocoeffs,(2,nindep,nbasis))
#                 else:
#                     beta = False
#                     self.logger.info("Creating attributes aonames[], mocoeffs[][]")
#                     self.aonames = []
#                     self.mocoeffs = Numeric.zeros((1,nindep,nbasis),"float")
# 
#                 base = 0
#                 for base in range(0,nindep,5):
#                     
#                     if self.progress:
#                         step=inputfile.tell()
#                         if step!=oldstep and random.random() < fupdate:
#                             self.progress.update(step,"Coefficients")
#                             oldstep=step
#                             
#                     colmNames = inputfile.next()
#                     symmetries = inputfile.next()
#                     eigenvalues = inputfile.next()
#                     for i in range(nbasis):
#                         line = inputfile.next()
#                         if base==0 and not beta: # Just do this the first time 'round
#                             # Changed below from :12 to :11 to deal with Elmar Neumann's example
#                             parts = line[:11].split()
#                             if len(parts)>1: # New atom
#                                 atomname = "%s%s" % (parts[2],parts[1])
#                             orbital = line[11:20].strip()
#                             self.aonames.append("%s_%s" % (atomname,orbital))
# 
#                         part = line[21:].replace("D","E").rstrip()
#                         temp = [] 
#                         for j in range(0,len(part),10):
#                             temp.append(float(part[j:j+10]))
#                         if beta:
#                             self.mocoeffs[1,base:base+len(part)/10,i] = temp
#                         else:
#                             self.mocoeffs[0,base:base+len(part)/10,i] = temp
                 
        inputfile.close()

        if self.progress:
            self.progress.update(nstep,"Done")
            
        if hasattr(self,"geovalues"): self.geovalues = Numeric.array(self.geovalues,"f")
        if hasattr(self,"scfenergies"): self.scfenergies = Numeric.array(self.scfenergies,"f")
        if hasattr(self,"scfvalues"): self.scfvalues = [Numeric.array(x,"f") for x in self.scfvalues]

        

# Note to self: Needs to be added to the main parser
    def extractTrajectory(self):
        """Extract trajectory information from a Gaussian logfile."""
        inputfile = open(self.filename,"r")
        self.traj = []
        self.trajSummary = []
        for line in inputfile:
            if line.find(" Cartesian coordinates:")==0:
                coords = []
                for i in range(self.natom):
                    line = inputfile.next()
                    parts = line.strip().split()
                    # Conversion from a.u. to Angstrom
                    coords.append([ self.float(x)*0.5292 for x in [parts[3],parts[5],parts[7]] ])
                self.traj.append(coords)
            if line==" Trajectory summary\n":
                # self.trajSummaryHeader = inputfile.next().strip().split()
                header = inputfile.next()
                line = inputfile.next()
                while line.find("Max Error")==-1:
                    parts = line.strip().split("  ")
                    self.trajSummary.append(parts)
                    line = inputfile.next()
        inputfile.close()
        assert len(self.traj)==len(self.trajSummary)
        
if __name__=="__main__":
    import doctest,adfparser
    doctest.testmod(adfparser,verbose=False)
