'''
Created on Nov 14, 2012

@author: Erics
'''
import re
import random
import util
if __name__ == '__main__':
    running = True
    menu = True
    while running:
        if menu:
            print "What would you like to do?"
            print "1.  Roll some dice"
            print "2.  Normal Game"
            print "3.  Crazy Game"
            print "4.  Probability Tables"
            print "5.  Cumulative Results"
            print "0.  Exit"
        menuChoice = raw_input("Choice: ")
        try:
            menuChoice = abs(int(menuChoice))
        except ValueError:
            print 'Choice must be one of the options'
            menuChoice = 0
        if menuChoice == 1:
            print "Enter a dice roll like this: 2d6+3"
            print "To run multiple throws type: 5 2d6+3"
            print "When running multiple throws, the total sum and average value of all rolls are shown by default,"
            print "each individual throw is shown up to 100 throws."
            print "each dice roll is shown up to 20 rolls"
            print "Type 0 to go back to the main menu."
            rolling = True
            while rolling:
                numThrows = 1
                numDice = 1
                die = 1
                modifier = 0
                print " "
                roll = raw_input("Roll: ")
                try:
                    input = int(roll)
                    if input == 0: rolling = False
                except ValueError:
                    pass
                if rolling == True:
                    if re.search(' ', roll) != None:
                        numThrows,diceType = re.split(' ',roll, 1)
                        try:
                            numThrows = abs(int(numThrows))
                        except ValueError:
                            print 'Number of throws must be an integer'
                            numThrows = 1
                    else:
                        numThrows = 1
                        diceType = roll
                    
                    if re.search('d|D', diceType) != None:
                        numDice, die = re.split('d|D', diceType, 1)
                    else:
                        die = diceType
                    try:
                        numDice = abs(int(numDice))
                    except ValueError:
                        numDice = 1
                        print 'Number of dice must be an integer'
                        
                    if len(str(numThrows)) > 6:
                        print "Too many throws, throwing dice 1,000,000 times (the maximum)"
                        numThrows = 1000000
                    if len(str(numDice)) > 6:
                        print "Too many dice, rolling 1,000,000 dice (the maximum)"
                        numDice = 1000000
                    if (numThrows * numDice) > 10000000:
                        numThrows = 10000000/numDice
                        print "Too high a total number of throws*dice, throwing ",numDice," dice the maximum number of times: ",numThrows
        
                    if re.search('[-]', die) != None:
                        die, modifier = re.split('[-]', die)
                        try:
                            modifier = -1 * int(modifier)
                        except ValueError:
                            print 'Only an integer can follow a plus or minus sign'
                            modifier = 0
                    elif re.search('[+]', die) != None:
                        die, modifier = re.split('[+]', die)
                        try:
                            modifier = int(modifier)
                        except ValueError:
                            print 'Only an integer can follow a plus or minus sign'
                            modifier = 0
                    try:
                        die = abs(int(die))
                    except ValueError:
                        print 'size of die must be an integer'
                        die = 0
                    if die > 1000000:
                        print 'A die must have no more than 1,000,000 sides'
                        die = 1000000
                    
                    t = 0
                    a = 0
                    throws = []
                    allRolls = []
                    for i in range(numThrows):
                        rolls = []
                        tt = 0
                        for j in range(numDice):
                            roll = random.randint(1, die)
                            tt += roll
                            rolls.append(roll)
                        tt += modifier
                        t += tt
                        allRolls.append(rolls)
                        throws.append(tt)
                    a = round(t/float(len(throws)), 2)
                    
                    print "Result:"
                    if numThrows == 1:
                        if numDice <= 20:
                            print throws[0],",     Rolls: ",allRolls[0],"+",modifier
                        else:
                            print throws[0]
                    else:
                        if len(str(throws)) <= 100:
                            rollNumber = 0
                            for i in throws:
                                rollNumber +=1
                                if rollNumber < 10: spaces = "  "
                                elif rollNumber < 100: spaces = " "
                                else: spaces = ""
                                if numDice <= 20:
                                    print rollNumber,":",spaces,i,",     Rolls: ",allRolls[rollNumber-1],"+",modifier
                                else:
                                    print rollNumber,":",spaces,i
                        print "Total:   ",t
                        print "Average: ",a
        elif menuChoice == 2:
            print "How many times do you want to play this round? (type 0 for Main Menu)"
            playing = True
            while playing:
                numGames = 1
                numDice = 1
                die = 1
                modifier = 0
                print " "
                numGames = raw_input("Games: ")
                try:
                    numGames = int(numGames)
                    if numGames == 0: playing = False
                except ValueError:
                    print "The number of times you want to play must be an integer between 1 and 1000"
                    numGames = 0
                    playing = False
                if numGames > 1000 or numGames < 1:
                    print "The number of times you want to play must be an integer between 1 and 1000"
                    numGames = 0
                    playing = False
                    
                if playing == True:
                    print "What type of game?  Type '0' for random otherwise use '2d6+3' format"
                    diceType = raw_input("Rolls: ")
                    try:
                        diceType = abs(int(diceType))
                    except ValueError:
                        pass
                    if diceType == 0:
                        print 'Playing',numGames," random games."
                        print 'Number ranges will all fall within -1,000,000 to 1,000,000'
                        
                        dNDDice = [4, 6, 8, 10, 12, 20]
                        smallDice = range(2, 51)
                        for i in range(21, 52): smallDice.append(i)
                        mediumDice = range(51, 256)
                        largeDice = range(256, 10000)
                        roundTenDice = range(30, 10000, 10)
                        roundHundredDice = range(100, 10000, 100)
                        
                        score = 0
                        results = []
                        for attempt in range(numGames):
                            diceDist = util.Counter()
                            diceDist[1] = 5
                            for number in range(1, 11): diceDist[number] = 10*(1/float(len(range(2,10))))
                            for number in range(11, 101): diceDist[number] = 3*(1/float(len(range(11,100))))
                            for number in range(101, 1001): diceDist[number] = 1*(1/float(len(range(101,1000))))
                            diceDist.normalize()
                            numDice = util.chooseFromDistribution( diceDist )
    
                            sizeDist = util.Counter()
                            for number in smallDice: sizeDist[number] = 150*(1/float(len(smallDice)))
                            for number in mediumDice: sizeDist[number] = 40*(1/float(len(mediumDice)))
                            for number in largeDice: sizeDist[number] = 10*(1/float(len(largeDice)))
                            for number in dNDDice: sizeDist[number] += 350*(1/float(len(dNDDice)))
                            for number in roundTenDice: sizeDist[number] += 40*(1/float(len(roundTenDice)))
                            for number in roundHundredDice: sizeDist[number] += 60*(1/float(len(roundTenDice)))
                            
                            sizeDist.normalize()
                            diceSize = util.chooseFromDistribution( sizeDist )
                            
                            
                            modDist = util.Counter()
                            modDist[0] = 150
                            if diceSize <= 50:
                                
                                for number in smallDice: modDist[number] = 44*(1/float(len(smallDice)))
                                for number in mediumDice: modDist[number] = 10*(1/float(len(mediumDice)))
                                for number in largeDice: modDist[number] = 2*(1/float(len(largeDice)))
                                for number in dNDDice: modDist[number] += 44*(1/float(len(dNDDice)))
                                for number in roundTenDice: sizeDist[number] += 5*(1/float(len(roundTenDice)))
                                for number in roundHundredDice: sizeDist[number] += 5*(1/float(len(roundTenDice)))
                            elif diceSize <= 300:
                                for number in smallDice: modDist[number] = 28*(1/float(len(smallDice)))
                                for number in mediumDice: modDist[number] = 29*(1/float(len(mediumDice)))
                                for number in largeDice: modDist[number] = 15*(1/float(len(largeDice)))
                                for number in dNDDice: modDist[number] += 28*(1/float(len(dNDDice)))
                                for number in roundTenDice: modDist[number] += 5*(1/float(len(roundTenDice)))
                                for number in roundHundredDice: modDist[number] += 5*(1/float(len(roundTenDice)))
                            else:
                                for number in smallDice: modDist[number] = 23*(1/float(len(smallDice)))
                                for number in mediumDice: modDist[number] = 23*(1/float(len(mediumDice)))
                                for number in largeDice: modDist[number] = 31*(1/float(len(largeDice)))
                                for number in dNDDice: modDist[number] += 23*(1/float(len(dNDDice)))
                                for number in roundTenDice: modDist[number] += 10*(1/float(len(roundTenDice)))
                                for number in roundHundredDice: modDist[number] += 10*(1/float(len(roundTenDice)))
                            modDist.normalize()
                            modifier = util.chooseFromDistribution( modDist )
                            if random.randint(1, 100) > 90: modifier = -modifier
                            
                            lowest = (numDice+modifier)
                            highest = (numDice*diceSize+modifier)
                            
                            print " "
                            print "Choose a random number between",lowest,"and",highest
                            guess = raw_input("Guess: ")
                            try:
                                guess = int(guess)
                            except ValueError:
                                print "Fail, you did not enter an integer"
                                guess = lowest
                            if guess < lowest or guess > highest:
                                print "Fail, guess did not lie between",lowest,"and",highest
                                guess = lowest
                            actual = random.randint(lowest, highest)
                            prob = (highest-lowest+1)
                            error = abs(actual-guess)
                            
                            
                            bottomError = (guess - error)
                            if bottomError < lowest: bottomError = lowest
                            topError = (guess + error)
                            if topError > highest: topError = highest
                            errorRange = topError - bottomError
                            closeScore = 5*(.5 - (errorRange / float(prob)))
                            
                            print" "
                            print "Guess:         ",guess
                            print "Actual:        ",actual
                            #print "Error:         ",error
                            #print "% Chance Win:  ",'%.2g' % round((1/float(prob)), 8)
                            #print "% Chance Error:",'%.2g' % round((errorRange / float(prob)), 8)
                            prevScore = score
                            if error == 0:
                                score += prob
                                print 'Awesome!  You guessed right, a 1 in',prob,'chance!'
                            else:
                                score -= 1
                                print 'Sorry, you were off by',error,'out of',prob
                            if closeScore > 0:
                                print 'You were close, ','%.2g' % closeScore,'bonus points!'
                            else:
                                print 'You were really off, lose','%.2g' % closeScore,'points'
                                
                            score+=closeScore
                            print 'You have','%.2g' % score,'points'
                            results.append([guess, actual, error, prob, '%.2g' % (score - prevScore)])
                        print " "
                        print " "
                        print " "
                        for i in results:
                            print i
                        print " "
                        print "Final Score:",'%.2g' % score
                        if score > 0: print "You Win!"
                        if score < 0: print "You Lose!"
                            
                            
                        
                         
                        
                        
                        
                        
                    
                    
                    
                    
                    else:
                        if re.search('d|D', diceType) != None:
                            numDice, die = re.split('d|D', diceType, 1)
                        else:
                            die = diceType
                        try:
                            numDice = abs(int(numDice))
                        except ValueError:
                            numDice = 1
                            print 'Number of dice must be an integer'
                            
                        if len(str(numThrows)) > 6:
                            print "Too many throws, throwing dice 1,000,000 times (the maximum)"
                            numThrows = 1000000
                        if len(str(numDice)) > 6:
                            print "Too many dice, rolling 1,000,000 dice (the maximum)"
                            numDice = 1000000
                        if (numThrows * numDice) > 10000000:
                            numThrows = 10000000/numDice
                            print "Too high a total number of throws*dice, throwing ",numDice," dice the maximum number of times: ",numThrows
            
                        if re.search('[-]', die) != None:
                            die, modifier = re.split('[-]', die)
                            try:
                                modifier = -1 * int(modifier)
                            except ValueError:
                                print 'Only an integer can follow a plus or minus sign'
                                modifier = 0
                        elif re.search('[+]', die) != None:
                            die, modifier = re.split('[+]', die)
                            try:
                                modifier = int(modifier)
                            except ValueError:
                                print 'Only an integer can follow a plus or minus sign'
                                modifier = 0
                        try:
                            die = abs(int(die))
                        except ValueError:
                            print 'size of die must be an integer'
                            die = 0
                        if die > 1000000:
                            print 'A die must have no more than 1,000,000 sides'
                            die = 1000000

    pass